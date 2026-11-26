"""Operations script, not part of the method: gather every recorded v2 result on the pod into a
few small text files and write them to stdout as a tar archive. Read-only on the pod; nothing
is written there. `v2_sync_results.sh` pipes this file to the pod's Python and unpacks the
archive into the repository's `results/` folder.

Files in the archive:
    README.md                      tables: validation scores, benchmark scores, training pairs
    RESULTS.tsv                    the schedule's own record of benchmark scores
    configs/X.yaml                 the configuration each setting ran with
    validation/X-roundR.json       each validation result as written by `run_optimised.py validate`
    pairs/X-roundR.json            each round's pairs summary
    lanes/laneG.plan, laneG.log    the job queues and their logs
"""
import io
import json
import sys
import tarfile
import time
from pathlib import Path

V = Path("/workspace/grace-work/v2")
S = Path("/workspace/grace-ops/sched-v2")
DATASETS = ["MER2023", "MER2024", "MELD", "IEMOCAPFour", "CMUMOSI", "CMUMOSEI", "SIMS", "SIMSv2", "OVMERDPlus"]
files = {}


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def score_cells(line):
    cells = [c.strip() for c in line.strip().split("&")][1:]
    return cells if len(cells) == len(DATASETS) + 1 else None


val_rows, pair_rows, score_rows = [], [], []
settings = sorted(d for d in V.iterdir() if (d / "optimised.yaml").is_file()) if V.is_dir() else []
sft = None
for d in settings:
    files[f"configs/{d.name}.yaml"] = (d / "optimised.yaml").read_text()
    for f in sorted(d.glob("validation/round*/validation.json")):
        text = f.read_text()
        v = json.loads(text)
        files[f"validation/{d.name}-round{v['round']}.json"] = text
        s = v["scores"]
        cells = [f"{s[t]['ew_f1']:.2f} ± {s[t]['sem']:.2f}" for t in ("all", "ovlabel", "description")]
        cells += [f"{s['all']['labels_per_answer']:.2f}", (v.get("sha256") or "")[:12]]
        if v["round"] == 0:
            sft = ["SFT epoch 30", 0, *cells]
        else:
            val_rows.append([d.name, v["round"], *cells])
    for f in sorted(d.glob("round*/pairs-summary.json")):
        text = f.read_text()
        p = json.loads(text)
        r = f.parent.name.replace("round", "")
        files[f"pairs/{d.name}-round{r}.json"] = text
        q, m = p.get("by_question_type", {}), p.get("by_method", {})
        pair_rows.append([d.name, r, p.get("pairs"), q.get("ovlabel"), q.get("description"), m.get("ranking", 0)])
if sft:
    val_rows.insert(0, sft)

results = V / "RESULTS.tsv"
if results.is_file():
    files["RESULTS.tsv"] = results.read_text()
    for line in results.read_text().splitlines():
        p = line.split("\t")
        cells = score_cells(p[5]) if len(p) >= 6 else None
        if cells:
            score_rows.append([p[1], *cells, p[3][:12], p[0]])

for f in sorted(S.glob("lane*.plan")) + sorted(S.glob("lane*.log")):
    files[f"lanes/{f.name}"] = f.read_text()

readme = "# GRACE v2 results from the pod\n\nWritten by `ops/v2_results_report.py` at " \
         + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + ". Do not edit by hand.\n\n"
readme += "## Validation (500 held-out clips, 4 answers per clip, wheel F1 x100, mean ± SEM)\n\n"
readme += table(["setting", "round", "all", "ovlabel", "description", "labels per answer", "checkpoint sha256"], val_rows) \
    if val_rows else "none yet\n"
readme += "\n## Benchmark (MER-UniBench, unchanged evaluation)\n\n"
readme += table(["run", *DATASETS, "mean", "checkpoint sha256", "recorded"], score_rows) if score_rows else "none yet\n"
readme += "\n## Training pairs\n\n"
readme += table(["setting", "round", "pairs", "ovlabel", "description", "ranking pairs"], pair_rows) if pair_rows else "none yet\n"
files["README.md"] = readme

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tar:
    for name, text in sorted(files.items()):
        data = text.encode("utf-8")
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))
sys.stdout.buffer.write(buf.getvalue())
