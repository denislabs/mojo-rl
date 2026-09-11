# +--------------------------------------------------------------------------+ #
# | Asset packs: the declaration, the credential decision, and the pull
# +--------------------------------------------------------------------------+ #
"""Gate `mojo_rl/assets/` — §9's infrastructure.

    pixi run build-http                              # ONCE
    pixi run mojo run -I . tests/assets/test_asset_packs.mojo

Hermetic: the pack is built here, served by
`tools/io/mock_monitor_server.py` on a loopback port, and pulled back. Nothing
needs the network, a credential, or a live service.

## What could be wrong, and what each check is for

* **A malformed declaration resolves anyway.** A pack with no `sha256` would
  download and be unverifiable; one with a climbing `dest` would write outside
  the env. Both are refused at parse time, which is the cheapest place to find
  them.
* **⚠⚠ THE WRONG CREDENTIAL IS SENT.** This is the whole reason `provider=`
  exists. Two `https://` URLs can need `HF_TOKEN`, `RL_MONITOR_API_KEY`, or
  nothing — and getting it wrong either 401s or, far worse, sends a secret to a
  host that should never have seen one.
* **The bytes arrive corrupt.** A pack that extracts to the wrong meshes is
  worse than one that fails to download.
* **A second pull re-downloads.** Then `assets-pull` stops being a preflight
  someone runs and becomes a cost they learn to skip.
* **The materialised tree is a symlink.** It breaks under rsync and the moment
  the cache is cleared — the same failure `project-promote` avoids.
"""

from std.os import getenv, setenv

from mojo_rl.core.dotenv import load_dotenv
from std.os.path import exists
from std.time import sleep

from mojo_rl.assets.pack import Pack, load_packs, parse_packs
from mojo_rl.assets.resolve import (
    archive_path,
    extract_dir,
    pack_status,
    pull_pack,
    resolve_url,
)
from mojo_rl.io.fileio import remove_file, write_text_atomic
from mojo_rl.io.http import http_shim_available
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.io.sha256 import sha256_file


comptime PORT_FILE = "/tmp/mojo_rl_pack_gate_port"
comptime LOG_FILE = "/tmp/mojo_rl_pack_gate_log"
comptime WORK = "/tmp/mojo_rl_pack_gate"


def _start_server() raises -> String:
    for p in [String(PORT_FILE), String(LOG_FILE)]:
        try:
            remove_file(p)
        except:
            pass
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py " + String(PORT_FILE) + " "
        + String(LOG_FILE) + " 180 > /tmp/mojo_rl_pack_gate_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(PORT_FILE):
            var f = open(String(PORT_FILE), "r")
            var port = String(f.read().strip())
            f.close()
            if port.byte_length() > 0:
                return "http://127.0.0.1:" + port
        sleep(0.1)
    raise Error("the mock server never wrote " + String(PORT_FILE))


def _refused(text: String, what: String) raises -> Bool:
    """Whether `parse_packs` rejects a declaration."""
    try:
        _ = parse_packs(text, what, String("/tmp"))
        return False
    except:
        return True


def main() raises:
    print("=== asset packs (§9) ===")

    # ⚠⚠ THE GATE OWNS ITS CACHE, AND IT LEARNED THIS THE HARD WAY. Running
    # against `~/.cache/mojo_rl` meant a MUTATION SWEEP polluted it: the H10
    # mutant disables the sha256 check, so it cached a deliberately-wrong
    # archive AND marked it `.ok` — after which the next honest run found the
    # marker, skipped the fetch entirely, and PASSED check 7b without
    # verifying anything. A gate that shares mutable state with its own
    # previous runs can be made vacuous by them.
    _ = run_capture("rm -rf " + quote_arg(String(WORK)))
    _ = setenv("MOJO_RL_CACHE", String(WORK) + "/cache", True)

    if not http_shim_available():
        raise Error("the HTTP shim is not built — `pixi run build-http`")
    var checks = 0

    # ── 1. a declaration round-trips ────────────────────────────────
    var good = (
        "schema_version=1\n"
        "pack=unitree_g1@v1\nprovider=hf\n"
        "url=https://hf.co/datasets/O/a/resolve/main/g1.tar.zst\n"
        "sha256=" + String("9" * 64) + "\nbytes=19923456\n"
        "dest=assets/unitree_g1\n"
        "pack=so_arm101@v2\nprovider=noeira\nid=so101-assets@v2\n"
        "sha256=" + String("a" * 64) + "\nbytes=16781312\n"
        "dest=assets/so_arm101\ntoken_env=MY_OWN_TOKEN\n"
    )
    var pf = parse_packs(good, String("gate"), String("mojo_rl/envs/robots"))
    if len(pf.packs) != 2:
        raise Error("expected 2 packs, parsed " + String(len(pf.packs)))
    var a = pf.packs[0].copy()
    var b = pf.packs[1].copy()
    if a.ref() != "unitree_g1@v1" or a.provider != "hf":
        raise Error("first pack wrong: " + a.ref() + " " + a.provider)
    if b.ref() != "so_arm101@v2" or b.id != "so101-assets@v2":
        raise Error("second pack wrong: " + b.ref() + " " + b.id)
    if b.token_env != "MY_OWN_TOKEN":
        raise Error("token_env lost")
    # ⚠ A RECORD ENDS AT THE NEXT `pack=`. If the grammar leaked, the second
    # pack would have inherited the first's url.
    if b.url.byte_length() != 0:
        raise Error("the second record inherited the first's url: " + b.url)
    var again = parse_packs(pf.render(), String("gate2"), String("x"))
    if len(again.packs) != 2 or again.packs[1].token_env != "MY_OWN_TOKEN":
        raise Error("render/parse is not a fixpoint")
    print("  declaration: 2 packs, records do not bleed, render round-trips")
    checks += 3

    # ── 2. a declaration that cannot resolve is REFUSED ─────────────
    var bad = [
        # no sha256 — would download and be unverifiable
        ("pack=x@v1\nprovider=https\nurl=http://h/x.tar\ndest=assets/x\n", "no sha256"),
        # short sha256
        ("pack=x@v1\nprovider=https\nurl=http://h/x.tar\nsha256=abc\ndest=assets/x\n", "short sha256"),
        # no dest — nothing would know where to put it
        ("pack=x@v1\nprovider=https\nurl=http://h/x.tar\nsha256=" + String("9"*64) + "\n", "no dest"),
        # ⚠ a climbing dest would write outside the env
        ("pack=x@v1\nprovider=https\nurl=http://h/x.tar\nsha256=" + String("9"*64) + "\ndest=../../etc\n", "climbing dest"),
        ("pack=x@v1\nprovider=https\nurl=http://h/x.tar\nsha256=" + String("9"*64) + "\ndest=/etc/passwd\n", "absolute dest"),
        # unknown provider — would fail on a box at 3am, not in the editor
        ("pack=x@v1\nprovider=ftp\nurl=http://h/x.tar\nsha256=" + String("9"*64) + "\ndest=assets/x\n", "unknown provider"),
        # noeira with a url instead of an id
        ("pack=x@v1\nprovider=noeira\nurl=http://h/x.tar\nsha256=" + String("9"*64) + "\ndest=assets/x\n", "noeira without id"),
        # hf with no url
        ("pack=x@v1\nprovider=hf\nsha256=" + String("9"*64) + "\ndest=assets/x\n", "hf without url"),
    ]
    var refused = 0
    for c in bad:
        if _refused(c[0], String("gate")):
            refused += 1
        else:
            print("    ACCEPTED: " + c[1])
    print(
        "  refusals: " + String(refused) + " of " + String(len(bad))
        + " malformed declarations refused"
    )
    if refused != len(bad):
        raise Error(String(len(bad) - refused) + " malformed packs accepted")
    checks += 1

    # ── 3. ⚠⚠ THE CREDENTIAL FOLLOWS THE PROVIDER ───────────────────
    #
    # The reason `provider=` exists at all. Sending the wrong secret either
    # 401s or hands a credential to a host that should never see one.
    var p_https = Pack(String("x"))
    p_https.provider = String("https")
    p_https.url = String("http://h/x.tar")
    p_https.sha256 = String("9" * 64)
    p_https.dest = String("assets/x")
    if resolve_url(p_https).bearer.byte_length() != 0:
        raise Error("provider=https sent a credential to an anonymous URL")

    var p_hf = p_https.copy()
    p_hf.provider = String("hf")
    # ⚠ THE SAME SOURCE THE RESOLVER USES: environment FIRST, then `.env`.
    # Asserting against `getenv` alone would have this gate fail on every
    # machine where the token lives in a dotenv — which is every machine here.
    var hf_tok = getenv("HF_TOKEN")
    if hf_tok == "":
        try:
            var env = load_dotenv(String(".env"))
            if "HF_TOKEN" in env:
                hf_tok = env["HF_TOKEN"]
        except:
            pass
    var got_hf = resolve_url(p_hf).bearer
    # ⚠ ASSERTED AGAINST THE ENVIRONMENT, not against a constant. On a box with
    # no HF_TOKEN a public pack must still resolve — §9's whole point is that a
    # clone works for anyone — so "no token" is correct there and only there.
    if hf_tok != "" and got_hf.byte_length() == 0:
        raise Error("provider=hf did not pick up HF_TOKEN")
    if hf_tok == "" and got_hf.byte_length() != 0:
        raise Error("provider=hf invented a token from nowhere")
    print(
        "  credential: https sends none; hf "
        + ("sends HF_TOKEN" if got_hf else "resolves anonymously (none set)")
    )

    # token_env overrides, and a missing one is an error rather than silence
    var p_tok = p_https.copy()
    p_tok.token_env = String("MOJO_RL_PACK_GATE_TOKEN")
    var raised = False
    try:
        _ = resolve_url(p_tok)
    except:
        raised = True
    if not raised:
        raise Error(
            "token_env named an unset variable and resolve_url carried on —"
            " the request would have gone out unauthenticated"
        )
    print("  credential: token_env with an empty variable is refused, not ignored")
    checks += 3

    # ── 4. a real pull: build, serve, fetch, verify, extract, link ──
    _ = run_capture(
        "mkdir -p " + quote_arg(String(WORK) + "/src/meshes") + " "
        + quote_arg(String(WORK) + "/env")
    )
    # two files, one nested, so extraction structure is actually checked
    _ = run_capture(
        "head -c 4096 /dev/urandom > " + quote_arg(String(WORK) + "/src/base.stl")
        + " && head -c 2048 /dev/urandom > "
        + quote_arg(String(WORK) + "/src/meshes/arm.stl")
    )
    var sha_base = sha256_file(String(WORK) + "/src/base.stl")
    var sha_arm = sha256_file(String(WORK) + "/src/meshes/arm.stl")
    _ = run_capture(
        "tar -czf " + quote_arg(String(WORK) + "/pack.tar.gz") + " -C "
        + quote_arg(String(WORK) + "/src") + " ."
    )
    var archive_sha = sha256_file(String(WORK) + "/pack.tar.gz")

    var base = _start_server()
    # The fixture stores a PUT and serves it back at the same key, which makes
    # it a perfectly good static host for this.
    _ = run_capture(
        "curl -s -X PUT --upload-file " + quote_arg(String(WORK) + "/pack.tar.gz")
        + " " + base + "/r2/packs/gate.tar.gz > /dev/null"
    )

    var kv = (
        "schema_version=1\npack=gate_pack@v1\nprovider=https\n"
        "url=" + base + "/r2/packs/gate.tar.gz\n"
        "sha256=" + archive_sha + "\ndest=assets/gate\n"
    )
    write_text_atomic(String(WORK) + "/env/assets.kv", kv)
    var loaded = load_packs(String(WORK) + "/env/assets.kv")
    var pack = loaded.packs[0].copy()

    if pack_status(pack, loaded.dir) != "absent":
        raise Error("the gate started with the pack already present")

    var st = pull_pack(pack, loaded.dir)
    if st != "pulled":
        raise Error("first pull returned '" + st + "'")
    var dest = loaded.dir + "/assets/gate"
    # ⚠ THE BYTES, not merely the presence of files.
    if sha256_file(dest + "/base.stl") != sha_base:
        raise Error("base.stl does not match what went into the pack")
    if sha256_file(dest + "/meshes/arm.stl") != sha_arm:
        raise Error("the NESTED mesh does not match — extraction flattened it")
    # ⚠ cache bookkeeping must not land in an env's asset directory
    if exists(dest + "/.ok"):
        raise Error("the .ok cache marker was materialised into the env")
    print("  pull: 2 files including a nested one, byte-identical, no .ok leaked")
    checks += 3

    # ── 5. ⚠⚠ HARD LINK, NEVER A SYMLINK — THE DIRECTORY AND THE FILES
    #
    # ⚠ BOTH ARE CHECKED, and the first version checked only the file. A
    # `ln -s <cache> <dest>` makes the DIRECTORY a link while every path
    # through it — `dest/base.stl` — is an ordinary file, so a file-only check
    # passes for exactly the defect it was written to catch. The mutation sweep
    # found this; reading it would not have.
    var link_dir = run_capture(
        "test -L " + quote_arg(dest) + " && echo YES || true", 1 << 12
    ).find("YES") >= 0
    if link_dir:
        raise Error(
            "the materialised asset DIRECTORY is a symlink into the cache — it"
            " breaks under rsync and the moment the cache is cleared"
        )
    var link_file = run_capture(
        "test -L " + quote_arg(dest + "/base.stl") + " && echo YES || true",
        1 << 12,
    ).find("YES") >= 0
    if link_file:
        raise Error("a materialised FILE is a symlink into the cache")
    print("  materialised: neither the directory nor its files are symlinks")
    checks += 2

    # ── 6. ⚠ a second pull moves NO bytes ───────────────────────────
    var before = run_capture(
        "grep -c GET " + quote_arg(String(LOG_FILE)) + " 2>/dev/null || echo 0",
        1 << 12,
    ).strip()
    if pull_pack(pack, loaded.dir) != "present":
        raise Error("a satisfied pull did not report 'present'")
    var after = run_capture(
        "grep -c GET " + quote_arg(String(LOG_FILE)) + " 2>/dev/null || echo 0",
        1 << 12,
    ).strip()
    if String(before) != String(after):
        raise Error(
            "a second pull made requests (" + String(before) + " -> "
            + String(after) + "); assets-pull must be free when satisfied"
        )
    print("  second pull: 'present', and the server saw no new request")
    checks += 2

    # ── 7a. a corrupted cache is REPAIRED, not served ───────────────
    #
    # ⚠ THE CORRECT BEHAVIOUR IS RECOVERY, NOT AN ERROR, and the first version
    # of this check asserted the opposite. A cached archive that fails its hash
    # is re-fetched and the run continues — which is what makes the cache safe
    # to keep. The property to assert is that the BYTES END UP RIGHT.
    _ = run_capture("rm -rf " + quote_arg(dest))
    _ = run_capture(
        "rm -rf " + quote_arg(extract_dir(pack)) + " && head -c 512 /dev/urandom > "
        + quote_arg(archive_path(pack))
    )
    if pull_pack(pack, loaded.dir) != "pulled":
        raise Error("a corrupted cache did not trigger a re-pull")
    if sha256_file(archive_path(pack)) != archive_sha:
        raise Error("the repaired cache still does not match its declaration")
    if sha256_file(dest + "/meshes/arm.stl") != sha_arm:
        raise Error("after repair the materialised bytes are still wrong")
    print("  a corrupted cache is refetched and repaired, and the bytes end right")
    checks += 2

    # ── 7b. ⚠⚠ BUT BYTES THAT NEVER MATCH ARE REFUSED ───────────────
    #
    # The hash is the only thing standing between a bad or substituted download
    # and a mesh that silently is not the one declared. A host serving
    # something else must fail the pull, not materialise it.
    var liar = pack.copy()
    liar.name = String("liar_pack")
    liar.sha256 = String("b" * 64)  # will never match what the fixture holds
    liar.dest = String("assets/liar")
    var liar_dest = loaded.dir + "/assets/liar"
    var caught = False
    var liar_said = String("")
    try:
        liar_said = pull_pack(liar, loaded.dir)
    except:
        caught = True
    if not caught:
        raise Error(
            "a pack whose bytes do not match its declared sha256 was accepted"
            " (pull_pack returned '" + liar_said + "')"
        )
    if exists(liar_dest):
        raise Error(
            "a refused pack still materialised files at " + liar_dest
            + " — a failed verification must leave nothing behind"
        )
    print("  bytes that do not match the declaration are refused, and land nothing")
    checks += 2

    _ = run_capture(
        "curl -s -X POST " + base + "/__shutdown > /dev/null 2>&1 || true"
    )
    _ = run_capture("rm -rf " + quote_arg(String(WORK)))
    print("[PASS] asset packs (" + String(checks) + " checks)")


# MUTANTS THIS FILE WAS CHECKED AGAINST (each must turn it red):
#   H1  validate() skips the sha256 length check   -> check 2
#   H2  validate() allows a climbing dest          -> check 2
#   H3  validate() accepts any provider            -> check 2
#   H4  a record does not reset at `pack=`         -> check 1
#   H5  provider=https sends HF_TOKEN              -> check 3
#   H6  token_env missing is silently ignored      -> check 3
#   H7  materialise symlinks the whole directory   -> check 4 (the .ok leaks
#                                                     THROUGH the link)
#   H7b the same, with the .ok removed so check 4   -> check 5, alone
#       cannot fire
#
# ⚠⚠ H7 SURVIVED THE FIRST SWEEP TWICE OVER, and both causes are worth having
# written down. The MUTANT was wrong: `ln -s src dest` where `dest` had already
# been created as a directory fails, and the `|| cp -a` fallback then did the
# right thing — so it tested nothing. And the CHECK was wrong: it looked at
# `dest/base.stl`, which is an ordinary file even when `dest` itself is a
# symlink. A file-only assertion passes for precisely the defect it names.
#   H8  materialise copies the .ok marker          -> check 4
#   H9  pull_pack ignores an existing dest         -> check 6
#   H10 fetch_to_cache skips the sha256 check      -> check 7b
