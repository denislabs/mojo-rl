"""`assets` — declare, check and fetch env asset packs.

    pixi run assets-status              # what is here, what is not
    pixi run assets-pull                # fetch everything declared
    pixi run assets-pull robots         # just one env's packs
    pixi run assets-pull --force        # re-fetch even if present

⚠⚠ NOTHING IS MIGRATED YET. This walks `mojo_rl/**/assets.kv`, of which there
are currently NONE — the meshes and textures are still tracked in git. That is
deliberate (`docs/PROJECT_LAYER_PLAN.md` §9): the infrastructure exists first,
so moving an env's assets out becomes a reversible decision made one env at a
time rather than a single commit nobody can undo.

⚠ `assets-pull` IS A PREFLIGHT, AND THAT IS ITS WHOLE POINT. A vast.ai box must
fetch packs BEFORE a run starts, not discover a missing mesh forty minutes into
training. Lazy fetch on first env construction is the fallback, not the plan.
"""

from std.sys import argv
from std.os.path import exists

from mojo_rl.assets.pack import Pack, PackFile, load_packs
from mojo_rl.assets.resolve import (
    archive_path,
    cache_root,
    pack_status,
    pull_pack,
    resolve_url,
)
from mojo_rl.io.fileio import file_size
from mojo_rl.io.proc import quote_arg, run_capture
from mojo_rl.utils.fmt import pad_left, pad_right


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _positional(n: Int) -> String:
    var av = argv()
    var seen = 0
    for i in range(1, len(av)):
        var a = String(av[i])
        if a.startswith("--"):
            continue
        if seen == n:
            return a
        seen += 1
    return String("")


def _find_pack_files(filter: String) raises -> List[String]:
    """Every `assets.kv` under `mojo_rl/`, optionally narrowed by a substring.

    ⚠ FOUND BY WALKING, NOT BY A LIST IN THIS FILE. An env that declares packs
    and is not listed somewhere central would silently never be fetched, and
    the failure would be a missing mesh at training time.
    """
    var out = List[String]()
    var txt = run_capture(
        "find mojo_rl -name assets.kv -type f 2>/dev/null | sort", 1 << 20
    )
    for line in txt.split("\n"):
        var s = String(line.strip())
        if s.byte_length() == 0:
            continue
        if filter.byte_length() > 0 and s.find(filter) < 0:
            continue
        out.append(s)
    return out^


def _mb(n: Int) -> String:
    return String(n // 1_000_000) + " MB"


def cmd_status() raises:
    var files = _find_pack_files(_positional(1))
    if len(files) == 0:
        print("no assets.kv anywhere under mojo_rl/")
        print(
            "  Nothing has been migrated yet — env assets are still tracked in"
            " git. See docs/PROJECT_LAYER_PLAN.md §9."
        )
        return

    var total = 0
    var present = 0
    var cached = 0
    var absent = 0
    var bytes_here = 0
    for f in files:
        var pf = load_packs(String(f))
        print(String(f))
        for i in range(len(pf.packs)):
            var p = pf.packs[i].copy()
            total += 1
            var st = pack_status(p, pf.dir)
            if st == "present":
                present += 1
            elif st == "cached":
                cached += 1
            else:
                absent += 1
            var a = archive_path(p)
            if exists(a):
                bytes_here += file_size(a)
            print(
                "  " + pad_right(st, 8) + pad_right(p.ref(), 22)
                + pad_right(p.provider, 9) + pad_left(_mb(p.bytes), 8)
                + "  -> " + p.dest
            )
    # ⚠ PRINT THE SCANNED COUNT BESIDE THE MISSING ONE. "0 absent" is also what
    # a walk that found no packs at all reports.
    print()
    print(
        String(total) + " pack(s) across " + String(len(files)) + " file(s): "
        + String(present) + " present, " + String(cached) + " cached, "
        + String(absent) + " absent"
    )
    print("cache " + cache_root() + " holds " + _mb(bytes_here))


def cmd_pull() raises:
    var files = _find_pack_files(_positional(1))
    var force = _has("--force")
    if len(files) == 0:
        print("no assets.kv anywhere under mojo_rl/ — nothing to pull")
        return
    var pulled = 0
    var already = 0
    var failed = 0
    for f in files:
        var pf = load_packs(String(f))
        for i in range(len(pf.packs)):
            var p = pf.packs[i].copy()
            try:
                var st = pull_pack(p, pf.dir, force)
                if st == "present":
                    already += 1
                    print("  present  " + p.ref())
                else:
                    pulled += 1
                    print("  pulled   " + p.ref() + " -> " + pf.dir + "/" + p.dest)
            except e:
                # ⚠ ONE PACK'S FAILURE DOES NOT STOP THE REST. A preflight that
                # aborts on the first unreachable host leaves someone fetching
                # the others one at a time by hand.
                failed += 1
                print("  FAILED   " + p.ref() + ": " + String(e))
    print()
    print(
        String(pulled) + " pulled, " + String(already) + " already present, "
        + String(failed) + " failed"
    )
    if failed > 0:
        raise Error(
            String(failed) + " pack(s) could not be fetched — a run started now"
            " would fail on a missing mesh instead"
        )


def cmd_where() raises:
    """Print the URL a pack would be fetched from, and with what credential.

    ⚠ IT PRINTS WHETHER A TOKEN WOULD BE SENT, NEVER THE TOKEN. "which secret
    does this use" is the question someone debugging a 401 actually has, and it
    is answerable without ever putting a credential on a terminal.
    """
    var files = _find_pack_files(_positional(1))
    for f in files:
        var pf = load_packs(String(f))
        for i in range(len(pf.packs)):
            var p = pf.packs[i].copy()
            print(p.ref() + "  (" + p.provider + ")")
            try:
                var r = resolve_url(p)
                print("  url     " + r.url)
                print(
                    "  bearer  "
                    + (
                        "yes, from " + p.token_env
                        if p.token_env
                        else ("yes, HF_TOKEN" if r.bearer else "none")
                    )
                    + (
                        "  (presigned; the storage host sees no credential)"
                        if p.provider == "noeira"
                        else ""
                    )
                )
            except e:
                print("  unresolvable: " + String(e))


def main() raises:
    var cmd = _positional(0)
    if cmd == "status" or cmd == "":
        cmd_status()
    elif cmd == "pull":
        cmd_pull()
    elif cmd == "where":
        cmd_where()
    else:
        print("usage: assets_cli <status|pull|where> [env-filter] [--force]")
        print("  see the module docstring, or docs/PROJECT_LAYER_PLAN.md §9")
