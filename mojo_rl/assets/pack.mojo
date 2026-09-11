# +--------------------------------------------------------------------------+ #
# | Asset packs — where an env's meshes and textures actually live
# +--------------------------------------------------------------------------+ #
"""`assets.kv` — a declaration of packs an env needs, and where to get them.

    mojo_rl/envs/robots/assets.kv

        schema_version=1

        pack=unitree_g1@v1
        provider=hf
        url=https://huggingface.co/datasets/Org/mojo-rl-assets/resolve/main/unitree_g1.tar.zst
        sha256=9f1c...
        bytes=19923456
        dest=assets/unitree_g1

        pack=so_arm101@v1
        provider=monitor
        id=so_arm101-assets@v1
        sha256=3ab0...
        bytes=16781312
        dest=assets/so_arm101

## ⚠⚠ `provider=` IS NOT A URL SCHEME, IT IS A CREDENTIAL DECISION

A bare `url=` can only describe something anyone may fetch. The moment a pack
is private — on our own host rather than a public Hub — the resolver has to
know *which secret to send*, and no amount of URL parsing tells it that: two
`https://` URLs can need `HF_TOKEN`, `RL_MONITOR_API_KEY`, or nothing at all.

So the provider names the ACCESS PATTERN:

| provider   | locator | credential                  | how the bytes are reached |
|------------|---------|-----------------------------|---------------------------|
| `https`    | `url=`  | none, or `token_env=`       | a plain GET |
| `hf`       | `url=`  | `HF_TOKEN`, or the `hf` CLI | bearer on the GET |
| `monitor`  | `id=`   | `RL_MONITOR_API_KEY`        | catalog call -> **presigned** URL -> unauthenticated GET |

⚠ `monitor` IS THE PRIVATE ONE, AND IT IS THE ONLY SHAPE THAT NEVER SENDS A
SECRET TO THE STORAGE HOST. The API key goes to our Worker, which answers with
a presigned URL; the transfer itself carries no credential at all. That is the
same rule the dataset and artifact paths already follow, and it is why a
private pack is not simply "a url with a token bolted on".

⚠ `token_env=` IS THE ESCAPE HATCH, so a fourth host does not need a fourth
provider: `provider=https token_env=MY_CDN_TOKEN` sends that variable as a
bearer. A provider is for an access PATTERN; a token name is just a name.

## What is NOT decided here

⚠ NOTHING IS MIGRATED. This file defines the format and the resolver; the
meshes and textures are still tracked in git. Moving them is a separate,
deliberate act — see `docs/PROJECT_LAYER_PLAN.md` §9 — and the infrastructure
existing first is what makes that act reversible.

## Why `.kv` and not JSON

The same reasons as `run.kv` (§2): repeating keys in order, `grep`-ability, and
hand-editability after the fact. ⚠ A record begins at each `pack=` line and
every key after it belongs to that record, which is what lets one file
describe several packs without nesting.
"""

from ..core.kv import KvWriter, kv_lines


comptime SCHEMA_VERSION = "1"

comptime PROVIDER_HTTPS = "https"
comptime PROVIDER_HF = "hf"
comptime PROVIDER_MONITOR = "monitor"


struct Pack(Copyable, Movable):
    """One pack: what it is, where it is, and what it must hash to."""

    var name: String
    """`unitree_g1` — the pack's identity, without its version."""
    var version: String
    var provider: String
    var url: String
    """For `https` / `hf`. Empty for `monitor`."""
    var id: String
    """For `monitor`: the catalog id to resolve. Empty otherwise."""
    var sha256: String
    """⚠ OF THE ARCHIVE, and REQUIRED. It is the cache key, the corruption
    check, and the reason a re-fetch can be skipped. A pack without one would
    make every `assets-pull` a download."""
    var bytes: Int
    var dest: String
    """Where the extracted tree is materialised, RELATIVE TO THE `assets.kv`
    that declared it. Relative so the file can be moved with its env."""
    var token_env: String
    """Override the provider's default credential variable. See the header."""

    def __init__(out self, name: String = String("")):
        self.name = name
        self.version = String("v1")
        self.provider = String(PROVIDER_HTTPS)
        self.url = String("")
        self.id = String("")
        self.sha256 = String("")
        self.bytes = 0
        self.dest = String("")
        self.token_env = String("")

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.version = move.version^
        self.provider = move.provider^
        self.url = move.url^
        self.id = move.id^
        self.sha256 = move.sha256^
        self.bytes = move.bytes
        self.dest = move.dest^
        self.token_env = move.token_env^

    def ref(self) -> String:
        """`unitree_g1@v1`."""
        return self.name + "@" + self.version

    def validate(self) raises:
        """Refuse a record that cannot possibly resolve.

        ⚠ REFUSED HERE, NOT AT FETCH TIME. A pack whose `sha256` is missing
        would download happily and then be unverifiable; one whose provider is
        unknown would fail on a box at 3am rather than in the editor. The
        cheapest place to find this is the moment the file is read.
        """
        if self.name.byte_length() == 0:
            raise Error("asset pack: a record with no name")
        if self.sha256.byte_length() != 64:
            raise Error(
                "asset pack '" + self.ref() + "': sha256 must be 64 hex"
                " characters, got " + String(self.sha256.byte_length())
                + ". It is the cache key AND the corruption check, so it is"
                " not optional."
            )
        if self.dest.byte_length() == 0:
            raise Error(
                "asset pack '" + self.ref() + "': no dest=, so nothing would"
                " know where to put the files"
            )
        # ⚠ AN ABSOLUTE OR CLIMBING `dest` WOULD WRITE OUTSIDE THE ENV. The
        # same rule the monitor applies to an artifact path, for the same
        # reason: this string ends up concatenated onto a directory.
        if self.dest.startswith("/") or self.dest.find("..") >= 0:
            raise Error(
                "asset pack '" + self.ref() + "': dest must be a relative path"
                " with no '..' — got '" + self.dest + "'"
            )
        if self.provider == PROVIDER_MONITOR:
            if self.id.byte_length() == 0:
                raise Error(
                    "asset pack '" + self.ref() + "': provider=monitor needs"
                    " id= (the catalog id), not url= — the URL is presigned"
                    " per request and cannot be written down"
                )
        elif self.provider == PROVIDER_HF or self.provider == PROVIDER_HTTPS:
            if self.url.byte_length() == 0:
                raise Error(
                    "asset pack '" + self.ref() + "': provider="
                    + self.provider + " needs url="
                )
        else:
            raise Error(
                "asset pack '" + self.ref() + "': unknown provider '"
                + self.provider + "'. Known: https, hf, monitor."
            )


struct PackFile(Movable):
    """One `assets.kv` and the directory it was found in."""

    var dir: String
    """The directory containing the file. `dest` is relative to this."""
    var packs: List[Pack]

    def __init__(out self, dir: String = String("")):
        self.dir = dir
        self.packs = List[Pack]()

    def __init__(out self, *, deinit move: Self):
        self.dir = move.dir^
        self.packs = move.packs^

    def render(self) raises -> String:
        var w = KvWriter(String("asset packs"))
        w.add(String("schema_version"), String(SCHEMA_VERSION))
        for i in range(len(self.packs)):
            var p = self.packs[i].copy()
            w.comment(String(""))
            w.add(String("pack"), p.ref())
            w.add(String("provider"), p.provider)
            if p.url.byte_length() > 0:
                w.add(String("url"), p.url)
            if p.id.byte_length() > 0:
                w.add(String("id"), p.id)
            w.add(String("sha256"), p.sha256)
            w.add(String("bytes"), String(p.bytes))
            w.add(String("dest"), p.dest)
            if p.token_env.byte_length() > 0:
                w.add(String("token_env"), p.token_env)
        return w^.done()


def parse_packs(text: String, what: String, dir: String) raises -> PackFile:
    """Read an `assets.kv`.

    ⚠ A RECORD BEGINS AT EACH `pack=` AND ENDS AT THE NEXT ONE. A key before
    the first `pack=` is a file-level key (`schema_version`); a key after one
    belongs to it. That is the whole grammar, and it is why the format needs no
    nesting.
    """
    var out = PackFile(dir)
    var cur = Pack()
    var open_record = False

    for line in kv_lines(text, what):
        var k = line.key
        var v = line.value
        if k == "pack":
            if open_record:
                cur.validate()
                out.packs.append(cur.copy())
            cur = Pack()
            open_record = True
            var at = v.rfind("@")
            if at < 0:
                cur.name = v
            else:
                cur.name = String(v[byte=0:at])
                cur.version = String(v[byte = at + 1 :])
            continue
        if not open_record:
            continue  # schema_version and anything else file-level
        if k == "provider":
            cur.provider = v
        elif k == "url":
            cur.url = v
        elif k == "id":
            cur.id = v
        elif k == "sha256":
            cur.sha256 = v
        elif k == "bytes":
            try:
                cur.bytes = atol(v)
            except:
                cur.bytes = 0
        elif k == "dest":
            cur.dest = v
        elif k == "token_env":
            cur.token_env = v

    if open_record:
        cur.validate()
        out.packs.append(cur^)
    return out^


def load_packs(path: String) raises -> PackFile:
    var txt: String
    with open(path, "r") as fh:
        txt = fh.read()
    var cut = path.rfind("/")
    var dir = String(".") if cut < 0 else String(path[byte=0:cut])
    return parse_packs(txt, path, dir)
