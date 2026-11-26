#!/bin/sh
# Operations script on the pod, not part of the method. Run once while v2 C / CK8 / CK12 round 1
# is still training (started by v2_start.sh):
#   - creates the single-change settings on CK8 round 1's answers: each is GRACE v1 CK8 (score
#     75.39) plus exactly one of the four changes, on ovlabel clips (v2 CK8 = all four):
#       CK8-R    ranking pairs
#       CK8-P    count-penalised reward and label-count rule, on GRACE's pairs
#       CK8-S    segment weighting (shared_weight 0.2), on GRACE's pairs
#       CK8-N    1/(pairs of its clip) weights, on GRACE's pairs. No ovlabel clip has two GRACE
#                pairs, so its pairs and weights equal CK8's: it is a second run of CK8 through
#                the v2 trainer and shows the run-to-run spread, not an effect
#   - replaces each GPU's v2_chain.sh by a v2_lane.sh queue: the running round 1 is adopted, then
#     validation and benchmark evaluation of round 1 (GPU 0 also validates SFT epoch 30 =
#     round 0, the reference), one new setting, and last the setting's own round 2. Only v2
#     settings are run (user decision): no v1 export is validated.
# The training processes are not touched: only the waiting chain shells are replaced.
set -e
S=/workspace/grace-ops/sched-v2
OPS=/workspace/grace-ops
PY=/workspace/grace-env/bin/python
V=/workspace/grace-work/v2
SRC=/workspace/grace-work/abl/CK8/round1
pgrep -f "v2_lane.sh" > /dev/null && { echo "lanes are running already"; exit 1; }
for spec in "CK8-R ranking" "CK8-P penalty" "CK8-S segment" "CK8-N clipnorm"; do
  set -- $spec; X=$1; W=$V/$X
  mkdir -p $W
  [ -f $W/verify-cache.jsonl ] || cp $V/CK8/verify-cache.jsonl $W/verify-cache.jsonl
  $PY - "$W" "$2" <<'EOF'
import sys, yaml
from pathlib import Path
w, on = Path(sys.argv[1]), sys.argv[2].split(",")
cfg = yaml.safe_load(Path("/workspace/grace-work/v2/CK8/optimised.yaml").read_text())   # same values as v2 CK8
new = yaml.safe_load(Path("/workspace/grace-v2/optimised-training/optimised.yaml").read_text())
cfg["validation"] = new["validation"]
cfg["paths"]["work_dir"] = str(w)
cfg["pairs"]["verify_cache"] = str(w / "verify-cache.jsonl")
cfg["rounds"] = 1
cfg["pairs"]["ranking_types"] = ["ovlabel"] if "ranking" in on else []
cfg["pairs"]["penalty_types"] = ["ovlabel"] if "penalty" in on else []
cfg["pairs"]["segment_types"] = ["ovlabel"] if "segment" in on else []
cfg["pairs"]["clip_normalize_types"] = ["ovlabel"] if "clipnorm" in on else []
out = w / "optimised.yaml"
if out.exists() and yaml.safe_load(out.read_text()) != cfg:
    raise SystemExit(f"{out} exists with different settings")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(w.name, {k: cfg["pairs"][k] for k in ("ranking_types", "penalty_types", "segment_types", "clip_normalize_types")})
EOF
done
plan() { [ -f $S/lane$1.plan ] && { echo "lane$1.plan exists"; exit 1; }; cat > $S/lane$1.plan; }
plan 0 <<EOF
adopt C 1
validate C 1
validate C 0
eval C 1
round CK8-R 1 $SRC
validate CK8-R 1
eval CK8-R 1
round C 2
validate C 2
eval C 2
EOF
plan 1 <<EOF
adopt CK8 1
validate CK8 1
eval CK8 1
round CK8-P 1 $SRC
validate CK8-P 1
eval CK8-P 1
round CK8 2
validate CK8 2
eval CK8 2
EOF
plan 2 <<EOF
adopt CK12 1
validate CK12 1
eval CK12 1
round CK8-S 1 $SRC
validate CK8-S 1
eval CK8-S 1
round CK8-N 1 $SRC
validate CK8-N 1
eval CK8-N 1
round CK12 2
validate CK12 2
eval CK12 2
EOF
# the waiting chain shells go (their python children keep training); the lanes take over
for X in C CK8 CK12; do
  for pid in $(pgrep -f "v2_chain.sh $X "); do kill -9 $pid; done
done
for spec in "0 C" "1 CK8" "2 CK12"; do
  set -- $spec
  setsid nohup sh $OPS/v2_lane.sh $1 $2 > $S/lane$1.out 2>&1 < /dev/null &
done
sleep 3
pgrep -af "v2_lane.sh|v2_chain.sh|run_optimised.py round" | cut -c1-110
