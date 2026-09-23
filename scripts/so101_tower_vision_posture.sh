#!/usr/bin/env bash
# so101_tower vision student, series 2: does the HUMAN-POSTURE expert make a
# student that uses REAL frames?
#
#   bash scripts/so101_tower_vision_posture.sh                  # arms V H, seeds 1 2 3
#   ARMS="V H" SEEDS="1 2 3" bash scripts/so101_tower_vision_posture.sh
#   STAGES="build" bash scripts/so101_tower_vision_posture.sh    # compile + the yaw GPU test
#
# THE SETUP (all of it new since series 1, and shared by every arm): episodes
# start at the real folded rest (9d4429420), the brick's yaw is drawn
# (ce11b131f), and every store and eval uses the REAL follower's joint zero
# (`--joint-zero follower`, 05a6298d9) — every student here is meant for the
# real arm.
#
# THE ARMS
#   V  200 episodes of the DEFAULT (vertical) expert — the control
#   H  200 episodes of `--posture human` (b971ad6fc, 4635d325a)
#   G  400 episodes of `--posture human` — OFF by default: series 1 measured
#      400 vs 200 expert episodes at 49.2 +- 10.2 vs 52.3 +- 9.2 % (3 seeds)
# (120 clean + 80 noisy per 200, as series 1.) Each arm x seed: ACT on the
# host data path, then TWO scores:
#   sim   tower_act_eval, 128 held-out placements (seeds 30000+), follower zero
#   real  tower_real_check on the rig's undistorted cube-in-bowl import —
#         mean action error against the recorded teleop, and the `hold ens.`
#         baseline: a student that does not beat it ignores the real frames
#         (series-1 students: 40-47 deg, their own mean-action baseline).
#
# NEEDS on the box: the real store at $REAL (8.5 GB, scp from the laptop's
# ~/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320_undist.h5).
# Everything else is generated here.
#
# OUTPUT: build/so101_vision/posture/results.tsv, one row per run, and a
# per-arm summary. Resumable: every stage skips finished work. Run inside
# tmux, from the repo root.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PATH=/workspace/.pixi-bin/bin:$PATH:$HOME/.pixi/bin

ARMS="${ARMS:-V H}"
SEEDS="${SEEDS:-1 2 3}"
STAGES="${STAGES:-build demos render train}"
REAL="${REAL:-$HOME/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320_undist.h5}"
D=projects/so101-tower/demos
B=build/so101_vision/posture
BIN=build/so101_vision/bin
mkdir -p "$B" "$BIN"
RES=$B/results.tsv
[[ -s $RES ]] || printf 'arm\tseed\trun\tbest_val_l1\tsim_success\tsim_no_grasp\treal_mean_err\treal_hold_err\n' > "$RES"
log() { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
MEM_GB=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 64)

grep -q ':yaw' noeira/tasks/tasks/so101_tower_cube_in_bowl.task || {
    echo "cube_in_bowl does not draw the brick's yaw: this checkout predates ce11b131f"; exit 1; }
grep -q 'base_qpos_jitter' noeira/tasks/families/so101_tower.family || {
    echo "the family does not start folded: this checkout predates 9d4429420"; exit 1; }
[[ -d projects/so101-tower && -f projects/so101-tower/project.kv ]] && export ACT_PROJECT=so101-tower

build() {  # src out [flags...]
    local src=$1 out=$2; shift 2
    if [[ ! -x $out || $src -nt $out ]]; then
        pixi run -e nvidia mojo build -j 8 "$@" -I . -o "$out" "$src" > "$out.build.log" 2>&1 \
            || { echo "BUILD FAILED: $src"; grep -A6 'error:' "$out.build.log" | head -30; return 1; }
    fi
}
if has build; then
    log "build (one at a time: ${MEM_GB} GB RAM)"
    pixi install -e nvidia > /dev/null
    build examples/so101/tower_expert_record.mojo $BIN/tower_expert
    build examples/so101/tower_demo_rerender.mojo $BIN/tower_rerender
    build examples/so101/act_so101_train_gpu.mojo $BIN/act_train_host -D ACT_HOST_DATA
    build examples/so101/tower_act_eval.mojo $BIN/tower_act_eval
    build examples/so101/tower_real_check.mojo $BIN/tower_real_check
    log "the brick-yaw reset on CUDA (test_device_placement_gpu)"
    pixi run -e nvidia mojo run -I . tests/tasks/test_device_placement_gpu.mojo 2>&1 \
        | grep -v 'intercept\]' | tail -8
    log "build ok"
fi

VC=$D/p2_vertical_clean.demo;  VN=$D/p2_vertical_noisy.demo
HC=$D/p2_human_clean.demo;     HN=$D/p2_human_noisy.demo
if has demos; then
    log "demos (CPU, four in parallel)"
    [[ -s $VC ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 51000 \
        --quiet --out $VC > $B/gen_vc.log 2>&1 &
    [[ -s $VN ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 300 --seed 54000 \
        --noise 0.02 --flat-noise --quiet --out $VN > $B/gen_vn.log 2>&1 &
    # human: ~50 % clean, ~30 % with noise (measured) — enough for arm G's 240 + 160
    [[ -s $HC ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 600 --seed 41000 \
        --posture human --quiet --out $HC > $B/gen_hc.log 2>&1 &
    [[ -s $HN ]] || $BIN/tower_expert so101_tower_cube_in_bowl --episodes 700 --seed 44000 \
        --posture human --noise 0.02 --flat-noise --quiet --out $HN > $B/gen_hn.log 2>&1 &
    wait
    for f in $B/gen_*.log; do echo "$(basename $f): $(tail -n 2 $f | head -1)"; done
fi

store_of() { echo "$D/p2_$1.rendered.h5"; }
render_arm() {
    local out; out=$(store_of "$1")
    [[ -s $out ]] && return 0
    local demos caps
    case $1 in
        V) demos=$VC,$VN; caps=120,80 ;;
        H) demos=$HC,$HN; caps=120,80 ;;
        G) demos=$HC,$HN; caps=240,160 ;;
    esac
    $BIN/tower_rerender --demos "$demos" --per-file "$caps" --joint-zero follower \
        --out "$out.partial" 2>&1 | grep -v '^Warning: attached\|intercept\]' \
        | grep -v '^  *[0-9]* episodes |' > "$B/render_$1.log"
    tail -5 "$B/render_$1.log"
    mv "$out.partial" "$out"
}
if has render; then
    log "render arms: $ARMS (follower joint zero)"
    for a in $ARMS; do render_arm "$a" & done
    wait
    for a in $ARMS; do [[ -s $(store_of "$a") ]] || { echo "render of arm $a failed"; exit 1; }; done
fi

train_one() {
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    grep -q 'best validation l1' "$B/act_$tag.log" 2>/dev/null && return 0
    ACT_STORE=$(store_of "$a") ACT_SEED=$s ACT_NO_MONITOR=1 ACT_RESIDENT_GB=60 \
        $BIN/act_train_host > "$B/act_$tag.log" 2>&1 || { echo "train $tag FAILED"; tail -20 "$B/act_$tag.log"; return 1; }
    log "trained $tag"
}
score_one() {
    local a=$1 s=$2 tag="$1_s$2"
    grep -q "^$a	$s	" "$RES" && return 0
    local run; run=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 '^  run ' | awk '{print $2}')
    local val; val=$(tr '\r' '\n' < "$B/act_$tag.log" | grep -m1 'best validation l1' | awk '{print $4}')
    # the sim eval and the real check side by side (a GPU each is plenty)
    $BIN/tower_act_eval --ckpt "$run/checkpoints" --episodes 128 --joint-zero follower \
        > "$B/eval_$tag.log" 2>&1 &
    local pe=$!
    if [[ -s $REAL ]]; then
        $BIN/tower_real_check --ckpt "$run/checkpoints" --student-zero follower \
            --store "$REAL" --store-zero follower > "$B/real_$tag.log" 2>&1 \
            || { echo "real check $tag FAILED"; tail -20 "$B/real_$tag.log"; }
    fi
    wait $pe || { echo "eval $tag FAILED"; tail -20 "$B/eval_$tag.log"; return 1; }
    local ok ng; ok=$(grep -m1 'SUCCESS' "$B/eval_$tag.log" | awk '{print $2}')
    ng=$(grep -m1 'no grasp' "$B/eval_$tag.log" | awk '{print $3}')
    local rm="-" rh="-"
    if [[ -s $REAL ]]; then
        rm=$(grep -m1 '^RESULT' "$B/real_$tag.log" | grep -o 'l1_all=[^ ]*' | cut -d= -f2 || echo "-")
        rh=$(grep -m1 '^RESULT' "$B/real_$tag.log" | grep -o 'hold_ens=[^ ]*' | cut -d= -f2 || echo "-")
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$s" "$run" "$val" "$ok" "$ng" "$rm" "$rh" >> "$RES"
    log "$tag: sim $ok/128 | real err $rm (hold $rh) | $run"
}
if has train; then
    for s in $SEEDS; do
        for a in $ARMS; do
            log "train $a seed $s"
            train_one "$a" "$s" || continue
            score_one "$a" "$s" &
        done
    done
    wait
fi

log "results"
column -t -s $'\t' "$RES" || cat "$RES"
log "done"
