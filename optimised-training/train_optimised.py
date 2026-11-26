"""Optimised GRACE training: GRACE's train_grace.py path with SegmentTrainer.

Identical to GRACE / the Standard DPO baseline: official DPO config.yaml / blank_model / dpo
loss config, BF16, lr 5e-6, batch 8 with 8 accumulation steps, max_length 4096 without
truncation, RMSprop + warm-up, official worker_main, baseline PairData/OmniModel/OmniTrainer,
reference = verified SFT epoch 30, the same pair/annotation validation against AffectGPT.
Different (the method): SegmentPairData (segment and pair weights), trainer=SegmentTrainer,
the reference / post-training passes use the weighted per-token log-probs.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

import optimised_interface as oi  # noqa: F401  (puts GRACE and the baseline on sys.path)
from grace_loss import GraceLossConfig
from train_grace import read_grace_pairs, validate_against_original  # GRACE, unchanged

gi, base = oi.gi, oi.base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--reference-adapter", type=Path, required=True)
    ap.add_argument("--policy-checkpoint", type=Path, required=True)
    ap.add_argument("--policy-sha256")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--eval-pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exp-name", default="grace_optimised")
    ap.add_argument("--beta", type=float, required=True, help="Per-token-scale beta (weighted mean log-probs)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--n-eval-examples", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reference", choices=["live", "precomputed"], default="live")
    ap.add_argument("--loss-config", type=Path, required=True, help="JSON with GraceLossConfig fields")
    ap.add_argument("--shared-weight", type=float, required=True,
                    help="Weight of response tokens shared by chosen and rejected (1 = setting C)")
    ap.add_argument("--warmup-steps", type=int)
    args = ap.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())

    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Linux/Python 3.10, as the baseline does")
    if not math.isfinite(args.beta) or args.beta <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("beta and lr must be finite and positive")
    if not 0 <= args.shared_weight <= 1:
        raise ValueError("shared-weight must be in [0, 1]")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    loss_fields = json.loads(args.loss_config.read_text(encoding="utf-8"))
    if not loss_fields.get("length_normalize"):
        raise ValueError("The optimised loss always uses weighted per-token means: set loss.length_normalize: true")
    loss_cfg = GraceLossConfig(beta=args.beta, **loss_fields)
    rows, eval_rows = read_grace_pairs(args.pairs), read_grace_pairs(args.eval_pairs)
    validate_against_original(rows + eval_rows, args.dataset_root)
    if loss_cfg.nll_min_chosen_score == "median":
        scores = [r["r_chosen"] for r in rows if r.get("r_chosen") is not None]
        if not scores:
            raise ValueError("nll_min_chosen_score: median needs r_chosen in the pairs file")
        loss_cfg.nll_min_chosen_score = float(statistics.median(scores))
    elif isinstance(loss_cfg.nll_min_chosen_score, str):
        raise ValueError("nll_min_chosen_score must be null, a number or 'median'")
    margins = [r["margin"] for r in rows if r.get("margin") is not None]
    default_margin = sum(margins) / len(margins) if margins else 0.0

    args.output.mkdir(parents=True, exist_ok=False)
    base.add_sft_path()
    sys.path.insert(0, str(base.DPO))
    import torch
    from omegaconf import OmegaConf
    import preference_datasets
    import trainers
    import utils
    from omni_components import load_processor
    for module in (preference_datasets, trainers, utils):
        if Path(module.__file__).resolve().parent != base.DPO:
            raise RuntimeError(f"Module collision with official DPO: {module.__file__}")
    spec = importlib.util.spec_from_file_location("official_dpo_entrypoint", base.DPO / "train.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    # ---- configuration: identical construction to the baseline / GRACE ----
    cfg = OmegaConf.load(base.DPO / "config/config.yaml")
    cfg.pop("defaults")
    cfg.model = OmegaConf.load(base.DPO / "config/model/blank_model.yaml")
    cfg.loss = OmegaConf.load(base.DPO / "config/loss/dpo.yaml")
    cfg.loss.beta = args.beta
    cfg.model.name_or_path = cfg.model.tokenizer_name_or_path = str(args.snapshot)
    cfg.model.policy_dtype = cfg.model.reference_dtype = "bfloat16"
    cfg.datasets, cfg.trainer = ["affectgpt_preferences"], "SegmentTrainer"
    cfg.local_run_dir, cfg.local_dirs = str(args.output), [str(args.runtime / "cache")]
    cfg.exp_name = args.exp_name
    cfg.lr = args.lr
    if args.warmup_steps is not None:
        if args.warmup_steps < 0:
            raise ValueError("warmup-steps must not be negative")
        cfg.warmup_steps = args.warmup_steps
    cfg.max_length, cfg.max_prompt_length = args.max_length, args.max_length - 1
    cfg.n_eval_examples = args.n_eval_examples
    cfg.sample_during_eval = False
    cfg.activation_checkpointing = args.gradient_checkpointing
    cfg.batch_size, cfg.eval_batch_size = args.batch_size, args.eval_batch_size
    cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if cfg.batch_size % cfg.gradient_accumulation_steps:
        raise ValueError("batch_size must be divisible by gradient_accumulation_steps")
    if len(rows) % cfg.batch_size or cfg.eval_every % cfg.batch_size:
        raise ValueError("batch_size must divide pair count and eval_every; no samples may be silently dropped")
    if len(eval_rows) < cfg.eval_batch_size or cfg.n_eval_examples % cfg.eval_batch_size:
        raise ValueError("Supply complete preference-evaluation batches")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires one visible CUDA GPU with BF16 support")
    OmegaConf.resolve(cfg)
    if OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing official configuration fields: {OmegaConf.missing_keys(cfg)}")
    (args.runtime / "cache").mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = utils.get_local_dir(cfg.local_dirs)
    OmegaConf.save(cfg, args.output / "config.yaml")
    (args.output / "grace-loss.json").write_text(json.dumps(vars(loss_cfg), indent=2), encoding="utf-8")
    (args.output / "optimised-run.json").write_text(json.dumps({
        "pairs": str(args.pairs), "pairs_sha256": gi.sha256_file(args.pairs),
        "eval_pairs_sha256": gi.sha256_file(args.eval_pairs),
        "policy_checkpoint": str(args.policy_checkpoint),
        "policy_sha256": gi.sha256_file(args.policy_checkpoint),
        "reference_sha256": gi.sha256_file(args.reference_adapter),
        "reference_mode": args.reference, "shared_weight": args.shared_weight,
        "grace": str(oi.GRACE), "baseline": str(gi.BASELINE)}, indent=2), encoding="utf-8")

    processor = load_processor(str(args.snapshot))
    data = oi.SegmentPairData(rows, eval_rows, processor, args.dataset_root,
                              preference_datasets.tokenize_batch_element, default_margin,
                              shared_weight=args.shared_weight)
    if gi.sha256_file(args.reference_adapter) != gi.EPOCH30_RUNNER_SHA256:
        raise ValueError("The reference must be the verified SFT epoch-30 runner checkpoint")
    log_path = args.output / "grace-pair-log.jsonl"
    with oi.segment_interfaces(preference_datasets, trainers, data, loss_cfg, log_path):
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(args.snapshot), cache_dir=utils.get_local_dir(cfg.local_dirs))
        if tokenizer.pad_token_id is None:  # same rule as BasicTrainer.__init__
            tokenizer.pad_token_id = tokenizer.eos_token_id
        table = None
        if args.reference == "precomputed":
            reference = base.OmniModel(base.load_sft_model(args.snapshot, args.reference_adapter, trainable=False))
            utils.disable_dropout(reference)
            table = oi.precompute_segment_logps(trainers, preference_datasets, reference, tokenizer, cfg,
                                                args.output / "reference-logps.json")
            del reference
            torch.cuda.empty_cache()
            reference = oi.PrecomputedSegmentReference(table)
        else:
            reference = base.OmniModel(base.load_sft_model(args.snapshot, args.reference_adapter, trainable=False))
        policy = base.OmniModel(gi.load_policy(args.snapshot, args.policy_checkpoint,
                                               expected_sha256=args.policy_sha256, trainable=True))
        if cfg.activation_checkpointing:
            policy.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        utils.disable_dropout(policy)
        utils.disable_dropout(reference)
        official.worker_main(0, 1, cfg, policy, reference)

        # Post-training pass, as GRACE: sigma(beta*h) of the FINAL policy on every training
        # pair (with the loss's weighted means), read by the next round's replay.
        policy.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if table is None:
            table = oi.precompute_segment_logps(trainers, preference_datasets, reference, tokenizer, cfg,
                                                args.output / "reference-logps.json", splits=("train",))
        final = oi.precompute_segment_logps(trainers, preference_datasets, policy, tokenizer, cfg,
                                            args.output / "final-policy-logps.json", splits=("train",),
                                            tag="final-policy")
        pref = oi.preference_probabilities(final, table, args.beta)
        missing = [r["id"] for r in rows if r["id"] not in pref]
        if missing:
            raise RuntimeError(f"Post-training pass missed {len(missing)} pairs, e.g. {missing[0]}")
        tmp = args.output / "final-pref.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as stream:
            for r in rows:
                stream.write(json.dumps({"id": r["id"], "type": r.get("type", "R"),
                                         "pref_prob": pref[r["id"]]}) + "\n")
        tmp.replace(args.output / "final-pref.jsonl")


if __name__ == "__main__":
    main()
