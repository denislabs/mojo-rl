#!/usr/bin/env bash
# MEASURE THE IMPLEMENTATION/PRECISION RATIO ACROSS THE LIBERO FAMILIES.
#
#     bash scripts/libero_axis_sweep_cuda.sh                 # the control, then all families
#     FAMILIES="libero_living_room_scene3 libero_goal" bash scripts/libero_axis_sweep_cuda.sh
#     SKIP_CONTROL=1 bash scripts/libero_axis_sweep_cuda.sh  # only if the control already passed
#
# `libero_family_batched` check 4 now gates a RATIO, not a constant: the device's
# float32 step against a CPU float32 step (the IMPLEMENTATION axis) divided by
# the CPU float32 step against a CPU float64 one (the PRECISION axis, the floor
# no implementation can beat). `IMPL_OVER_PREC_MAX` is 10 and is PROVISIONAL —
# it was set from ONE family, where the worst per-lane ratio is 3.24. This
# measures the distribution that justifies it.
#
# ⚠⚠ COST. `FAMILY` is a compile-time constant, so every family is its own
# build, and the float32 leg added a SECOND full CPU physics instantiation. Budget
# 10-20 minutes PER FAMILY on a rented box — the full sweep is hours. Results are
# appended as they land and completed families are SKIPPED on a re-run, so this
# can be stopped and resumed.
set -uo pipefail

# ⚠⚠ LC_ALL=C, AND IT IS A CORRECTNESS FIX, NOT TIDINESS. Under a comma-decimal
# locale (this box's shell is fr_FR) `awk` does not recognise "3.2432" as a
# number, so every comparison falls back to STRING order and "3.2432" > "14.7".
# Measured on the synthetic fixture: the summary named 3.000 / scene3 as the
# worst when 14.7 / spatial was, and printed the mean as "5,667". `sort -g` has
# the same exposure. A sweep whose whole purpose is to find the WORST ratio must
# not be able to get that backwards.
export LC_ALL=C

cd "$(dirname "$0")/.."

OUT=${OUT:-build/diag/axis_sweep.tsv}
STEPS=${STEPS:-25}
CPU_LANES=${CPU_LANES:-5}
CTRL_FAMILY=${CTRL_FAMILY:-libero_living_room_scene3}
GATE=examples/libero/libero_family_batched.mojo

say() { printf '\n=== %s\n' "$*"; }

command -v pixi >/dev/null || { echo "pixi is not on PATH" >&2; exit 1; }

say "0. the LIBERO asset pack (meshes + textures are gitignored)"
if [ -d noeira/envs/libero/assets/stable_hope_objects ]; then
    echo "already materialised"
else
    pixi run assets-pull libero
fi
[ -d noeira/envs/libero/assets/stable_hope_objects ] || {
    echo "the asset pack did not materialise — see noeira/envs/libero/assets.kv" >&2
    exit 1; }

# ── run one family, print "<ratio>\t<impl>\t<prec>\t<lane>\t<lanes>\t<verdict>" ──
# ⚠ A NON-ZERO EXIT IS DATA, NOT AN ERROR: the gate RAISES when it fails, and a
# failing family is exactly what this sweep is looking for.
run_family() {
    local fam=$1 bound_override=${2:-}
    local tmpd tmp log
    tmpd=$(mktemp -d)
    tmp="$tmpd/fam_$fam.mojo"
    log="$tmpd/out.txt"
    if [ -n "$bound_override" ]; then
        sed -e "s/^comptime FAMILY = .*/comptime FAMILY = \"$fam\"/" \
            -e "s/^comptime IMPL_OVER_PREC_MAX: Float64 = .*/comptime IMPL_OVER_PREC_MAX: Float64 = $bound_override/" \
            "$GATE" > "$tmp"
        grep -q "IMPL_OVER_PREC_MAX: Float64 = $bound_override" "$tmp" \
            || { echo "the bound sed did not take" >&2; rm -rf "$tmpd"; return 2; }
    else
        sed "s/^comptime FAMILY = .*/comptime FAMILY = \"$fam\"/" "$GATE" > "$tmp"
    fi
    grep -q "comptime FAMILY = \"$fam\"" "$tmp" \
        || { echo "the FAMILY sed did not take" >&2; rm -rf "$tmpd"; return 2; }
    pixi run -e nvidia mojo run -I . "$tmp" \
        --steps "$STEPS" --cpu-lanes "$CPU_LANES" > "$log" 2>&1
    RUN_LOG=$(cat "$log")
    # the numbers, straight out of the gate's own report
    # ⚠ THE GATED QUANTITY IS THE **MATERIAL** RATIO — impl above
    # IMPL_MATERIAL_ABS on the SAME lane — so that is what this table sorts on.
    # The unfiltered per-lane ratio is carried beside it, because a lane
    # climbing toward materiality is worth seeing before it trips the gate.
    # ⚠ `awk`, not `sed`: the report line carries parentheses and a `|`, and
    # escaping those through sed twice is how the previous version of these
    # regexes silently produced "-" for every family after a rename.
    # the gate prints one `AXISROW key=value ...` line for exactly this — see
    # its comment in the gate. Position-independent, so rewording the human
    # report cannot silently empty this table again.
    kv() { sed -n "s/.*AXISROW .*[ ]$1=\\([^ ]*\\).*/\\1/p" "$log" | tail -1; }
    MAT=$(kv mat_ratio); MIMPL=$(kv mat_impl); MPREC=$(kv mat_prec)
    MLANE=$(kv mat_lane); MLANES=$(kv mat_lanes); RATIO=$(kv all_ratio)
    if grep -q "=== PASS —" "$log"; then
        VERDICT=PASS
    elif grep -q "^  FAIL:" "$log"; then
        VERDICT="FAIL($(grep -m1 '^  FAIL:' "$log" | cut -c10-60 | tr '\t' ' '))"
    else
        VERDICT="NO-VERDICT(build or launch failed)"
    fi
    rm -rf "$tmpd"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${MAT:--}" "${MIMPL:--}" \
        "${MPREC:--}" "${MLANE:--}" "${MLANES:--}" "${RATIO:--}" "$VERDICT"
}

# ── 1. THE ANTI-VACUITY CONTROL ───────────────────────────────────────────
# A gate nobody has seen FAIL is a decoration. Tighten the bound below the
# observed ratio; the gate MUST fail. If it passes, the whole sweep below is
# measuring nothing and the script stops.
if [ "${SKIP_CONTROL:-0}" != "1" ]; then
    say "1. anti-vacuity control on $CTRL_FAMILY at bound 1.0 — MUST FAIL"
    ctrl=$(run_family "$CTRL_FAMILY" 1.0)
    echo "   $ctrl"
    case "$ctrl" in
        *FAIL*) echo "   OK: the gate's failure path is reachable and reported." ;;
        *NO-VERDICT*)
            echo "   ⚠ the control produced no verdict — build or launch failed." >&2
            echo "   Fix that first; nothing below is meaningful." >&2
            exit 1 ;;
        *)  echo "   ⚠⚠ THE CONTROL PASSED AT BOUND 1.0. The observed ratio was" >&2
            echo "   3.24 when this was written, so a bound of 1.0 must fail." >&2
            echo "   The gate is not wired to its own numbers — STOPPING." >&2
            exit 1 ;;
    esac
else
    say "1. control SKIPPED (SKIP_CONTROL=1)"
fi

# ── 2. THE SWEEP ──────────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || printf 'family\tmat_ratio\tmat_impl\tmat_prec\tlane\tmat_lanes\tall_ratio\tverdict\n' > "$OUT"

if [ -n "${FAMILIES:-}" ]; then
    # shellcheck disable=SC2206
    fams=($FAMILIES)
else
    fams=()
    # ⚠ A GLOB, NOT `ls | sed`: `ls` here is proxied and prints sizes beside the
    # names, which silently yields an EMPTY family list.
    for f in noeira/envs/libero/families/libero_*.family; do
        [ -e "$f" ] || continue
        fams+=("$(basename "$f" .family)")
    done
fi
say "2. sweeping ${#fams[@]} families (steps $STEPS, cpu-lanes $CPU_LANES) -> $OUT"
echo "   budget 10-20 min PER FAMILY; stop and re-run to resume."

for fam in "${fams[@]}"; do
    if cut -f1 "$OUT" | grep -qx "$fam"; then
        echo "   [skip] $fam (already in $OUT)"
        continue
    fi
    printf '   [run ] %s ... ' "$fam"
    row=$(run_family "$fam")
    printf '%s\t%s\n' "$fam" "$row" >> "$OUT"
    echo "$row"
done

# ── 3. THE DISTRIBUTION, which is the point ───────────────────────────────
say "3. per-lane implementation/precision ratio, worst first"
awk -F'\t' 'NR>1' "$OUT" | sort -t"$(printf '\t')" -k2 -gr \
  | awk -F'\t' '{printf "%-30s mat_ratio %-12s impl %-13s prec %-13s lane %-3s all %-12s %s\n", $1,$2,$3,$4,$5,$7,$8}'
say "summary"
awk -F'\t' 'NR==1{next} $2!="-"{n++; s+=$2; if($2>mx){mx=$2; mf=$1}}
     END{ if(n) printf "  %d families with a calibratable ratio | mean %.3f | WORST %.3f (%s)\n", n, s/n, mx, mf;
          else print "  no calibratable ratios — check the log" }' "$OUT"
awk -F'\t' 'NR==1{next} /FAIL/{print "  FAIL " $1 " -> " $8}' "$OUT"
echo
echo "⚠ SET THE BOUND FROM THIS DISTRIBUTION, not from one family, and never to"
echo "  make a run green. A genuine device defect sits ORDERS above the float32"
echo "  floor, not 3-10x, so keep headroom over the worst honest ratio."
