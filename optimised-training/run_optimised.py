"""Optimised GRACE driver: GRACE's run_grace.py with two stages replaced.

  python run_optimised.py round  --config optimised.yaml --round 1
  python run_optimised.py eval   --config optimised.yaml --round 1   # baseline v2 MER-UniBench, unchanged
  python run_optimised.py status --config optimised.yaml
  python run_optimised.py reuse  --config optimised.yaml --round 1 --from /workspace/grace-work/abl/C/round1

Stages ids, candidates (K on-policy answers), labels (AffectGPT extractor + wheel F1), claims,
verify and export are GRACE's, called unchanged. `pairs` is ranking.build (changes 2 and 4)
and `train` is train_optimised.py (change 1). `reuse` copies round 1's sampled answers, labels
and claims from an earlier run that sampled the same clips from SFT epoch 30, so the optimised
method is compared with that setting on the same round-1 candidates.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
GRACE = Path(os.environ.get("GRACE_ROOT", HERE.parent / "vendor" / "grace")).resolve()
for path in (GRACE, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_grace as rg  # noqa: E402  (GRACE driver, unchanged)

REUSED = ("candidates.jsonl", "labels.jsonl", "claims.jsonl")


def stage_pairs(cfg, r, d):
    import build_pairs as bp
    import ranking
    replay = []
    if r > 1 and cfg["pairs"]["replay_max_fraction"] > 0:
        prev = rg.round_dir(cfg, r - 1)
        final = prev / "train" / "final-pref.jsonl"
        if not final.exists():
            raise SystemExit(f"{final} is missing; it is written by train_optimised.py")
        replay = bp.select_replay(bp.read_jsonl(prev / "pairs.jsonl"), bp.read_jsonl(final))
    summary = ranking.build(d, cfg["pairs"], cfg["train"]["batch_size"], cfg["seed"] + r, replay, bp=bp)
    print(json.dumps(summary, indent=2))
    eval_path = Path(cfg["paths"]["work_dir"]) / "eval-pairs.jsonl"
    if not eval_path.exists():   # the same 16 baseline preference-validation rows as GRACE
        import gzip
        src = Path(cfg["paths"]["baseline_eval_pairs"])
        opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt", encoding="utf-8") as stream:
            rows = [json.loads(l) for l in stream if l.strip()]
        bp.write_jsonl(eval_path, bp.eval_rows_from_baseline(rows))


def stage_train(cfg, r, d):
    p, t = cfg["paths"], cfg["train"]
    ckpt, sha = rg.policy_for(cfg, r)
    out = d / "train"
    if out.exists():
        shutil.move(str(out), str(d / f"train.interrupted.{len(list(d.glob('train.interrupted.*')))}"))
    loss_path = d / "grace-loss.json"
    loss_path.write_text(json.dumps(cfg["loss"]), encoding="utf-8")
    cmd = [sys.executable, HERE / "train_optimised.py", "--runtime", p["runtime"], "--dataset-root", p["dataset_root"],
           "--snapshot", p["snapshot"], "--reference-adapter", p["epoch30_checkpoint"],
           "--policy-checkpoint", ckpt, "--pairs", d / "pairs.jsonl",
           "--eval-pairs", Path(p["work_dir"]) / "eval-pairs.jsonl", "--output", out,
           "--exp-name", f"grace_optimised_round{r}", "--beta", t["beta"], "--lr", t["lr"],
           "--batch-size", t["batch_size"], "--gradient-accumulation-steps", t["gradient_accumulation_steps"],
           "--eval-batch-size", t["eval_batch_size"], "--n-eval-examples", t["n_eval_examples"],
           "--max-length", t["max_length"], "--reference", t["reference"], "--loss-config", loss_path,
           "--shared-weight", cfg["segment"]["shared_weight"]]
    warmup = rg.warmup_steps_for(t, d / "pairs.jsonl")
    if warmup is not None:
        cmd += ["--warmup-steps", warmup]
    if sha:
        cmd += ["--policy-sha256", sha]
    rg.run(cmd)


def reuse(cfg, r, source):
    """Round-1 candidates/labels/claims of an earlier run on the same clips (epoch-30 sampler)."""
    if r != 1:
        raise SystemExit("Only round 1 can reuse answers: later rounds sample from their own policy")
    source, d = Path(source), rg.round_dir(cfg, r)
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} is not empty")
    d.mkdir(parents=True, exist_ok=True)
    rg.stage_ids(cfg, r, d)
    if json.loads((d / "ids.json").read_text()) != json.loads((source / "ids.json").read_text()):
        raise SystemExit("The source round sampled different clips (seed / clips_per_round differ)")
    for name in REUSED:
        shutil.copy2(source / name, d / name)
    (d / "reused-from.json").write_text(json.dumps({"source": str(source)}, indent=2), encoding="utf-8")
    for stage in ("ids", "candidates", "labels", "claims"):
        (d / f".done-{stage}").write_text("ok\n")
    # verify: the shared verify cache already holds every claim of these answers; the stage
    # is left to run, so any missing verdict is computed instead of assumed.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["round", "eval", "status", "reuse"])
    ap.add_argument("--config", type=Path, default=HERE / "optimised.yaml")
    ap.add_argument("--round", type=int)
    ap.add_argument("--from", dest="source", type=Path)
    args = ap.parse_args()
    cfg = rg.load(args.config)
    rg.stage_pairs, rg.stage_train = stage_pairs, stage_train   # the two replaced stages
    if args.command == "round":
        rg.do_round(cfg, args.round)
    elif args.command == "eval":
        rg.do_eval(cfg, args.round)
    elif args.command == "reuse":
        reuse(cfg, args.round, args.source)
    else:
        for r in range(1, cfg["rounds"] + 1):
            d = rg.round_dir(cfg, r)
            print(f"round {r}: " + " ".join(s for s in rg.STAGES if (d / f".done-{s}").exists()))


if __name__ == "__main__":
    main()
