#!/bin/zsh
# Measure peak memory of a build command (whole process tree) on macOS.
#
# Polls every second:
#   - sum of RSS over the command's full process tree (the compiler may be a
#     child of pixi/mojo — watching only the parent reports ~nothing)
#   - global swap-used delta (on a 16 GB host the compiler's excess goes to
#     swap/compressed memory, invisible to RSS)
#
# Kill-guard: if tree RSS exceeds RSS_LIMIT_GB or swap grows by more than
# SWAP_LIMIT_GB, the whole tree is killed before the machine thrashes.
#
# Usage:
#   ./scripts/measure_build_mem.sh [-r RSS_LIMIT_GB] [-s SWAP_LIMIT_GB] -- <command...>
# Example:
#   ./scripts/measure_build_mem.sh -r 13 -s 8 -- \
#     pixi run -e apple mojo build -I . examples/foo.mojo -o /tmp/foo

set -u

RSS_LIMIT_GB=13
SWAP_LIMIT_GB=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r) RSS_LIMIT_GB="$2"; shift 2 ;;
    -s) SWAP_LIMIT_GB="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "no command given" >&2
  exit 2
fi

swap_used_mb() {
  # vm.swapusage: total = 2048.00M  used = 1234.50M  free = ...
  # (decimal separator is locale-dependent — keep the integer part only)
  sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+)[.,][0-9]+M.*/\1/'
}

tree_rss_kb() {
  # BFS over ppid links from the root pid; sum RSS (KB).
  local root=$1
  ps -axo pid=,ppid=,rss= | awk -v root="$root" '
    { pid[NR]=$1; ppid[NR]=$2; rss[NR]=$3 }
    END {
      want[root]=1
      changed=1
      while (changed) {
        changed=0
        for (i=1;i<=NR;i++)
          if (ppid[i] in want && !(pid[i] in want)) { want[pid[i]]=1; changed=1 }
      }
      total=0
      for (i=1;i<=NR;i++) if (pid[i] in want) total+=rss[i]
      print total
    }'
}

swap0=$(swap_used_mb)
start=$(date +%s)

"$@" &
ROOT=$!

peak_rss_kb=0
peak_swap_mb=0
killed=""

while kill -0 "$ROOT" 2>/dev/null; do
  rss=$(tree_rss_kb "$ROOT")
  swap=$(swap_used_mb)
  swap_delta=$(( swap - swap0 ))
  (( swap_delta < 0 )) && swap_delta=0
  (( rss > peak_rss_kb )) && peak_rss_kb=$rss
  (( ${swap_delta:-0} > peak_swap_mb )) && peak_swap_mb=$swap_delta

  if (( rss > RSS_LIMIT_GB * 1024 * 1024 )); then
    killed="RSS ${RSS_LIMIT_GB}GB"
  elif (( ${swap_delta:-0} > SWAP_LIMIT_GB * 1024 )); then
    killed="swap +${SWAP_LIMIT_GB}GB"
  fi
  if [[ -n "$killed" ]]; then
    echo "KILL-GUARD tripped ($killed) — killing build tree" >&2
    pkill -9 -P "$ROOT" 2>/dev/null
    kill -9 "$ROOT" 2>/dev/null
    break
  fi
  sleep 1
done

wait "$ROOT" 2>/dev/null
build_status=$?
elapsed=$(( $(date +%s) - start ))

echo "----------------------------------------"
echo "exit status   : $build_status ${killed:+(KILLED: $killed)}"
echo "elapsed       : ${elapsed}s"
echo "peak tree RSS : $(( peak_rss_kb / 1024 )) MB"
echo "peak swap +   : ${peak_swap_mb} MB"
echo "----------------------------------------"
exit $build_status
