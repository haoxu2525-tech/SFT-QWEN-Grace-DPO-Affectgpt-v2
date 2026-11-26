#!/bin/sh
# Operations script on the pod, not part of the method. One v2 setting on one GPU:
#   sh v2_chain.sh SETTING GPU SOURCE_ROUND1
# Round 1 reuses SOURCE_ROUND1's answers/labels/claims (same clips, SFT epoch 30), then
# `run_optimised.py round 1`; the GPU is released while this setting's own two eval slots (on
# the same GPU, sched-v2/SETTING) evaluate round 1; when round 1 is scored (or its evaluation
# failed 3 times) round 2 runs, then the slots evaluate round 2. Each setting's whole workflow
# stays on its own GPU. sched-v2/SETTING/gpuG.reserved is present while the chain trains.
X=$1; G=$2; SRC=$3
S=/workspace/grace-ops/sched-v2/$1
PY=/workspace/grace-env/bin/python
CODE=/workspace/grace-v2/optimised-training
CFG=/workspace/grace-work/v2/$X/optimised.yaml
W=/workspace/grace-work/v2/$X
log() { echo "$(date -u +%FT%TZ) $X/gpu$G $*" >> $S/train.log; }
reserve() { : > $S/gpu$G.reserved; sleep 20; }   # running eval units on G stop within POLL
release() { rm -f $S/gpu$G.reserved; }
round() {
  reserve; log "start round $1"
  if ( cd $CODE && CUDA_VISIBLE_DEVICES=$G GRACE_ROOT=/workspace/grace WANDB_MODE=offline PYTHONUNBUFFERED=1 \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY run_optimised.py round --config $CFG --round $1 >> $W/round$1.log 2>&1 ); then
    log "done round $1: $(tr -d '\n ' < $W/round$1/pairs-summary.json | cut -c1-200)"; release
  else
    log "FAILED round $1 (see $W/round$1.log)"; release; exit 1
  fi
}
[ -f $S/STOP ] && exit 0
if [ ! -f $W/round1/.done-ids ]; then
  ( cd $CODE && GRACE_ROOT=/workspace/grace $PY run_optimised.py reuse --config $CFG --round 1 --from $SRC >> $W/round1.log 2>&1 ) \
    || { log "FAILED reuse of $SRC"; exit 1; }
  log "reused round-1 answers from $SRC"
fi
[ -f $W/round1/.done-export ] || round 1
log "waiting for round 1's score before round 2"
while [ ! -s $W/eval1.scores.txt ]; do
  [ -f $S/STOP ] && exit 0
  for f in $W/eval1.final.fails $W/eval1.u*.fails; do
    [ -f "$f" ] && [ "$(cat "$f")" -ge 3 ] && { log "round 1 evaluation failed; going on with round 2"; break 2; }
  done
  sleep 60
done
[ -f $S/STOP ] && exit 0
[ -f $W/round2/.done-export ] || round 2
log "chain finished"
