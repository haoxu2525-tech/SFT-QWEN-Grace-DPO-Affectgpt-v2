#!/bin/sh
# Operations script on the pod, not part of the method. Sets up and starts v2 C / CK8 / CK12:
# one config per setting (the repo's optimised.yaml with its own work dir, k_ovlabel and verify
# cache). Each setting runs its whole workflow on its own GPU (C 0, CK8 1, CK12 2): its training
# chain plus two evaluation slots (v1's unchanged eval_slot.sh) with its own schedule folder
# sched-v2/SETTING and eval plan; finished evaluations go to /workspace/grace-work/v2/RESULTS.tsv.
set -e
S=/workspace/grace-ops/sched-v2
PY=/workspace/grace-env/bin/python
OPS=/workspace/grace-ops
mkdir -p $S /workspace/grace-work/v2
for spec in "C 0 4" "CK8 1 8" "CK12 2 12"; do
  set -- $spec; X=$1; K=$3; W=/workspace/grace-work/v2/$X
  mkdir -p $W
  [ -f $W/verify-cache.jsonl ] || cp /workspace/grace-work/verify-cache.jsonl $W/verify-cache.jsonl
  $PY - "$W" "$K" <<'EOF'
import sys, yaml
from pathlib import Path
w, k = Path(sys.argv[1]), int(sys.argv[2])
cfg = yaml.safe_load(Path("/workspace/grace-v2/optimised-training/optimised.yaml").read_text())
cfg["paths"]["work_dir"] = str(w)
cfg["pairs"]["verify_cache"] = str(w / "verify-cache.jsonl")
cfg["k_ovlabel"] = k
cfg["rounds"] = 2
out = w / "optimised.yaml"
if out.exists() and yaml.safe_load(out.read_text()) != cfg:
    raise SystemExit(f"{out} exists with different settings")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
EOF
done
export SCHED_RESULTS=/workspace/grace-work/v2/RESULTS.tsv
for spec in "C 0" "CK8 1" "CK12 2"; do
  set -- $spec; X=$1; G=$2; SX=$S/$X
  mkdir -p $SX/locks; rm -f $SX/STOP
  [ -f $SX/eval.plan ] || for r in 1 2; do
    echo "v2-$X-r$r /workspace/grace-work/v2/$X/eval$r /workspace/grace-work/v2/$X/round$r/export 1"
  done > $SX/eval.plan
  for SLOT in a b; do
    SCHED_DIR=$SX setsid nohup sh $OPS/eval_slot.sh $G $SLOT helper > $SX/slot-$SLOT.out 2>&1 < /dev/null &
  done
  setsid nohup sh $OPS/v2_chain.sh $X $G /workspace/grace-work/abl/$X/round1 > $SX/chain.out 2>&1 < /dev/null &
done
echo started
