#!/bin/sh
# Operations script on the pod, not part of the method. One GPU's queue of v2 jobs:
#   sh v2_lane.sh GPU OWNER        (OWNER = the setting whose two evaluation slots run on GPU)
# Runs the lines of sched-v2/laneGPU.plan in order, one at a time:
#   adopt    SETTING ROUND          wait for a round started earlier (v2_chain.sh); resume it if it stopped
#   round    SETTING ROUND [SOURCE] run_optimised.py round; round 1 first reuses SOURCE's answers
#   validate SETTING ROUND          run_optimised.py validate (held-out wheel F1; round 0 = SFT epoch 30)
#   eval     SETTING ROUND          free the GPU for OWNER's evaluation slots and wait for the score
# SETTING is a folder of /workspace/grace-work/v2 (config optimised.yaml), or v1:NAME for a GRACE
# v1 setting /workspace/grace-work/abl/NAME (config grace.yaml; validate only).
# sched-v2/OWNER/gpuGPU.reserved keeps the evaluation slots off the GPU from a computing job
# until the next eval job (or the end of the queue), so no evaluation unit is started and cut off.
# A failed job is logged and the queue goes on; sched-v2/STOP-lanes ends it before the next job.
G=$1; OWN=$2
S=/workspace/grace-ops/sched-v2; SO=$S/$OWN
PY=/workspace/grace-env/bin/python
CODE=/workspace/grace-v2/optimised-training
log() { echo "$(date -u +%FT%TZ) gpu$G $*" | tee -a $S/lane$G.log >> $SO/train.log; }
where() {   # sets W (work dir) and CFG for a setting
  case "$1" in
    v1:*) W=/workspace/grace-work/abl/${1#v1:}; CFG=$W/grace.yaml;;
    *)    W=/workspace/grace-work/v2/$1; CFG=$W/optimised.yaml;;
  esac
}
reserve() {   # evaluation units on this GPU stop within the slots' poll time
  [ -f $SO/gpu$G.reserved ] && return
  : > $SO/gpu$G.reserved; sleep 20
  n=0; while ls $SO/busy.$G.* > /dev/null 2>&1 && [ $n -lt 60 ]; do sleep 10; n=$((n+1)); done
}
release() { rm -f $SO/gpu$G.reserved; }
py() {
  ( cd $CODE && CUDA_VISIBLE_DEVICES=$G GRACE_ROOT=/workspace/grace WANDB_MODE=offline PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY run_optimised.py "$@" )
}
do_round() {   # SETTING ROUND [SOURCE]
  where $1
  [ -f $W/round$2/.done-export ] && { log "$1 round $2: done already"; return; }
  [ -f $CFG ] || { log "FAILED $1 round $2: no $CFG"; return; }
  reserve
  if [ "$2" = 1 ] && [ -n "$3" ] && [ ! -f $W/round1/.done-ids ]; then
    py reuse --config $CFG --round 1 --from $3 >> $W/round1.log 2>&1 || { log "FAILED $1: reuse of $3"; return; }
    log "$1: reused round-1 answers from $3"
  fi
  log "start $1 round $2"
  if py round --config $CFG --round $2 >> $W/round$2.log 2>&1; then
    log "done $1 round $2: $(tr -d '\n ' < $W/round$2/pairs-summary.json | cut -c1-200)"
  else
    log "FAILED $1 round $2 (see $W/round$2.log)"
  fi
}
do_adopt() {   # SETTING ROUND
  where $1
  while pgrep -f "run_optimised.py round --config $CFG --round $2" > /dev/null; do sleep 30; done
  if [ -f $W/round$2/.done-export ]; then
    log "done $1 round $2: $(tr -d '\n ' < $W/round$2/pairs-summary.json | cut -c1-200)"
  else
    log "$1 round $2 stopped before its export; resuming it"; do_round $1 $2
  fi
}
do_validate() {   # SETTING ROUND
  where $1
  V=$W/validation/round$2
  [ -s $V/validation.json ] && { log "$1 round $2 validation: done already"; return; }
  [ "$2" = 0 ] || [ -f $W/round$2/.done-export ] || { log "SKIPPED validation of $1 round $2: no export"; return; }
  reserve
  log "start validation of $1 round $2"
  mkdir -p $W/validation
  if py validate --config $CFG --round $2 >> $W/validation/round$2.log 2>&1; then
    log "validated $1 round $2: $($PY -c "import json,sys; s=json.load(open(sys.argv[1]))['scores']; print(' '.join('%s %.2f (+-%.2f, %d clips)' % (k, v['ew_f1'], v['sem'], v['clips']) for k, v in s.items()))" $V/validation.json)"
  else
    log "FAILED validation of $1 round $2 (see $W/validation/round$2.log)"
  fi
}
do_eval() {   # SETTING ROUND   (v2 settings)
  where $1
  RT=$W/eval$2
  [ -f $W/round$2/.done-export ] || { log "SKIPPED evaluation of $1 round $2: no export"; return; }
  grep -q "^v2-$1-r$2 " $SO/eval.plan 2>/dev/null || echo "v2-$1-r$2 $RT $W/round$2/export 1" >> $SO/eval.plan
  release
  log "waiting for the benchmark score of $1 round $2"
  while [ ! -s $RT.scores.txt ]; do
    [ -f $S/STOP-lanes ] && return
    for f in $RT.final.fails $RT.u*.fails; do
      [ -f "$f" ] && [ "$(cat "$f")" -ge 3 ] && { log "FAILED evaluation of $1 round $2 (3 attempts)"; return; }
    done
    sleep 60
  done
  log "scored $1 round $2: $(tail -1 $RT.scores.txt | awk -F'&' '{print $NF}' | tr -d ' ')"
}
log "lane started"
exec 3< $S/lane$G.plan
while read VERB X R SRC <&3; do
  case "$VERB" in ""|\#*) continue;; esac
  [ -f $S/STOP-lanes ] && { log "lane stopped (STOP-lanes)"; release; exit 0; }
  case "$VERB" in
    adopt) do_adopt $X $R;;
    round) do_round $X $R $SRC;;
    validate) do_validate $X $R;;
    eval) do_eval $X $R;;
    *) log "unknown job: $VERB $X $R";;
  esac
done
release
log "lane finished: every line of lane$G.plan was handled"
