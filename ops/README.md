# Pod operations (not part of the method)

Scripts used on the GRACE pod. Evaluation always uses v1's unchanged `ops/eval_slot.sh` (GRACE repo)
with a schedule folder per setting (`/workspace/grace-ops/sched-v2/SETTING`).
Results: `/workspace/grace-work/v2/RESULTS.tsv`; status: `sh v2_status.sh`.

| Script | |
|---|---|
| `v2_start.sh`, `v2_chain.sh` | first start of v2 C / CK8 / CK12 (`k_ovlabel` 4 / 8 / 12), one setting per GPU: round 1 reusing the v1 setting's round-1 answers, evaluation, round 2, evaluation. Replaced by the lanes below while round 1 was training |
| `v2_lane.sh` | one GPU's job queue (`sched-v2/laneG.plan`): `adopt` a running round, `round`, `validate` (held-out wheel F1), `eval` (free the GPU for the benchmark evaluation and wait for its score). Stop before the next job: `touch sched-v2/STOP-lanes` |
| `v2_ablation_start.sh` | creates the one-change-at-a-time settings CK8-R / CK8-RP / CK8-RPS on CK8 round 1's answers, writes the three lane plans (round-1 validation and evaluation, one new setting per GPU, reference validations of SFT epoch 30 and v1 exports, then each setting's own round 2) and swaps the chains for lanes without touching the running trainings |
| `v2_status.sh` | stages, validation score, benchmark units and score of every v2 setting; reference validations; each lane's last event |
