# GRACE v2 results from the pod

Written by `ops/v2_results_report.py` at 2026-11-26T12:30:46Z. Do not edit by hand.

## Validation (500 held-out clips, 4 answers per clip, wheel F1 x100, mean ± SEM)

| setting | round | all | ovlabel | description | labels per answer | checkpoint sha256 |
|---|---|---|---|---|---|---|
| SFT epoch 30 | 0 | 64.56 ± 1.00 | 68.10 ± 1.37 | 60.94 ± 1.41 | 3.23 |  |
| C | 1 | 65.47 ± 1.00 | 68.69 ± 1.34 | 62.17 ± 1.47 | 3.50 | baea7f58c3b4 |
| CK12 | 1 | 68.45 ± 0.98 | 72.44 ± 1.28 | 64.37 ± 1.45 | 3.56 | c43a068fa87a |
| CK8 | 1 | 67.37 ± 1.00 | 71.70 ± 1.29 | 62.92 ± 1.49 | 3.41 | ce496fd5a857 |

## Benchmark (MER-UniBench, unchanged evaluation)

none yet

## Training pairs

| setting | round | pairs | ovlabel | description | ranking pairs |
|---|---|---|---|---|---|
| C | 1 | 2352 | 1417 | 935 | 1417 |
| CK12 | 1 | 2848 | 1914 | 934 | 1914 |
| CK8 | 1 | 2752 | 1817 | 935 | 1817 |
