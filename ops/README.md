# Pod operations (not part of the method)

Scripts used on the GRACE pod to run v2 settings C / CK8 / CK12 (`k_ovlabel` 4 / 8 / 12), one
setting per GPU, each running its whole workflow there: round 1 reusing the v1 setting's round-1
answers, evaluation of round 1, round 2, evaluation of round 2. Evaluation uses v1's unchanged
`ops/eval_slot.sh` (GRACE repo) with a schedule folder per setting (`/workspace/grace-ops/sched-v2/SETTING`).
Results: `/workspace/grace-work/v2/RESULTS.tsv`; status: `sh v2_status.sh`; stop: `touch sched-v2/SETTING/STOP`.
