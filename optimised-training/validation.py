"""Held-out validation: emotion-wheel F1 of a round's exported policy on clips no round trains on.

Rounds and settings are otherwise chosen on MER-UniBench, the test benchmark itself. This gives
a score from the training corpus instead: the LAST `validation.clips` (500) clips of the seeded
clip order that `ids` uses. Rounds take their clips from the front of that order (10 rounds x
2,000 leave the last 11,256 of 31,256 unused), so no round of any setting with the same seed
ever samples or trains on them, and every setting is validated on the same clips.

Nothing new is scored or prompted: the clips keep their own AffectGPT question (description or
ovlabel, as in training), the answers are sampled by GRACE's unchanged candidates stage from the
round's exported checkpoint (`validation.k` answers per clip), and each answer gets GRACE's
unchanged label stage (AffectGPT extractor + per-sample emotion-wheel F1 against the clip's
ground-truth labels, score_labels.py). The validation score is the mean of that F1, x100:
first over a clip's answers, then over clips.

  python run_optimised.py validate --config optimised.yaml --round 1   # round 1's export
  python run_optimised.py validate --config optimised.yaml --round 0   # SFT epoch 30 (the start)

Output: <work_dir>/validation/round<r>/validation.json (plus ids, answers and labels). The
sampling is not seeded to repeat (temperature 1, as in training-time sampling), so two runs of
the same checkpoint differ slightly; `sem` in the output is the standard error over clips.
"""
import json
import math
from pathlib import Path
import random


def held_out_ids(all_ids, seed, n, rounds, clips_per_round):
    """The last n ids of the order `ids` draws rounds from (sorted, then shuffled with the seed)."""
    ids = sorted(all_ids)
    random.Random(seed).shuffle(ids)
    if n <= 0 or rounds * clips_per_round + n > len(ids):
        raise SystemExit(f"{rounds} rounds x {clips_per_round} clips leave fewer than {n} of {len(ids)} clips unused")
    return ids[-n:]


def summarise(candidates, labels):
    """Mean per-sample wheel F1 (x100): per clip over its answers, then over clips."""
    by_clip = {}
    for row in labels:
        by_clip.setdefault(row["id"], []).append(row)
    out = {}
    for key in ("all", "ovlabel", "description"):
        clips = [c for c in candidates if key in ("all", c.get("label_type")) and c["id"] in by_clip]
        if not clips:
            continue
        means = [sum(float(r["r_label"]) for r in by_clip[c["id"]]) / len(by_clip[c["id"]]) for c in clips]
        rows = [r for c in clips for r in by_clip[c["id"]]]
        m = sum(means) / len(means)
        var = sum((x - m) ** 2 for x in means) / (len(means) - 1) if len(means) > 1 else 0.0
        out[key] = {"clips": len(clips), "answers": len(rows), "ew_f1": 100 * m,
                    "sem": 100 * math.sqrt(var / len(means)),
                    "hit_rate": 100 * sum(bool(r["hit"]) for r in rows) / len(rows),
                    "labels_per_answer": sum(len(set(r.get("labels") or [])) for r in rows) / len(rows),
                    "gt_labels_per_clip": sum(len(set(by_clip[c["id"]][0].get("gt_labels") or [])) for c in clips) / len(clips)}
    return out


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def validate(cfg, r, rg):
    """Sample and score the validation clips with round r's export (r = 0: SFT epoch 30)."""
    import grace_interface as gi
    v = cfg.get("validation") or {}
    n, k = int(v.get("clips", 500)), int(v.get("k", 4))
    ckpt, sha = rg.policy_for(cfg, r + 1)   # the policy round r + 1 would start from = round r's export
    d = Path(cfg["paths"]["work_dir"]) / "validation" / f"round{r}"
    d.mkdir(parents=True, exist_ok=True)
    ids = held_out_ids((row["id"] for row in gi.base.read_pairs(cfg["paths"]["baseline_pairs"])),
                       cfg["seed"], n, int(cfg["rounds"]), int(cfg["clips_per_round"]))
    ids_path = d / "ids.json"
    if ids_path.exists() and json.loads(ids_path.read_text(encoding="utf-8")) != ids:
        raise SystemExit(f"{ids_path} holds different clips (validation.clips or seed changed)")
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    sampling = dict(cfg.get("sampling") or {}, processes=int(v.get("processes", 2)), remotes=[])
    if not (d / ".done-candidates").exists():
        # GRACE's candidates stage, unchanged; its own seeds (offset 700000) and this GPU only.
        rg.stage_candidates(dict(cfg, k=k, seed=cfg["seed"] + 700000, sampling=sampling), r + 1, d)
        (d / ".done-candidates").write_text("ok\n")
    if not (d / ".done-labels").exists():
        rg.stage_labels(cfg, r, d)          # GRACE's label stage, unchanged
        (d / ".done-labels").write_text("ok\n")
    candidates, labels = read_rows(d / "candidates.jsonl"), read_rows(d / "labels.jsonl")
    if [c["id"] for c in candidates] != ids or any(len(c["candidates"]) != k for c in candidates):
        raise SystemExit(f"{d}: the sampled answers do not cover the validation clips")
    result = {"round": r, "checkpoint": str(ckpt), "sha256": sha, "clips": n, "k": k,
              "scores": summarise(candidates, labels)}
    (d / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result
