"""
Run self-modeling evals on a mixed test set sampled across all benchmarks and perturbations.

100 examples randomly sampled from GSM8K, HumanEval, BBQ, WildGuardTest.
Each example gets a random perturbation from its domain.
E1-E9 use all benchmarks; E10 only uses BBQ.

Usage:
    uv run python scripts/experiments/run_self_modeling_mixed.py \
        --models claude-haiku-4-5-20251001 \
        --n-samples 100 \
        --seed 42

    # With vLLM:
    uv run python scripts/experiments/run_self_modeling_mixed.py \
        --models Qwen/Qwen2.5-7B-Instruct \
        --n-samples 100 \
        --seed 42 \
        --vllm-url http://HOST:PORT/v1
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

_GREEN = "\033[92m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

# Benchmark → domain → available perturbations (excluding controls)
BENCHMARK_CONFIGS = {
    "gsm8k": {
        "domain": "math",
        "perturbations": ["hint42", "boxed45", "hallucinated_units", "sycophancy_smaller"],
    },
    "humaneval": {
        "domain": "coding",
        "perturbations": ["loop_decoy_for_while_range", "print_statement", "try_except_block", "out_of_scope", "builtin_line_budget"],
    },
    "bbq": {
        "domain": "fairness",
        "perturbations": ["bias_test_with_uncertainty", "no_evaluation", "common_sense", "metadata_prime"],
    },
    "wildguardtest": {
        "domain": "safety",
        "perturbations": ["previous_harm", "assume_malicious", "json_format", "urgency_pressure"],
    },
}


def build_mixed_test_set(n_samples: int, seed: int) -> list[dict]:
    """Build a mixed test set with n_samples examples per eval.

    E1-E9: n_samples examples spread across all benchmarks (~n/4 each).
    E10 (BBQ only): n_samples BBQ examples (extra BBQ added if needed).

    Total eval instances = 10 * n_samples (e.g., 10 * 10 = 100).

    Returns list of dicts with keys: benchmark, domain, perturbation_id, example_idx, example.
    """
    from prompt_attribution.eval.benchmarks import get_benchmark

    rng = random.Random(seed)

    benchmarks = list(BENCHMARK_CONFIGS.keys())
    n_per_benchmark = n_samples // len(benchmarks)
    remainder = n_samples % len(benchmarks)

    test_set = []
    global_idx = 0
    n_bbq = 0  # track how many BBQ examples we have

    for i, bm_name in enumerate(benchmarks):
        cfg = BENCHMARK_CONFIGS[bm_name]
        n = n_per_benchmark + (1 if i < remainder else 0)

        try:
            bm = get_benchmark(bm_name)
            examples = bm.load_examples(n_samples=n, random_seed=seed + i)
        except Exception as e:
            logger.warning(f"  Skipping {bm_name}: {e}")
            continue

        for ex in examples:
            # Random perturbation for this example
            pert_id = rng.choice(cfg["perturbations"])

            test_set.append({
                "global_idx": global_idx,
                "benchmark": bm_name,
                "domain": cfg["domain"],
                "perturbation_id": pert_id,
                "example": ex,
            })
            global_idx += 1
            if bm_name == "bbq":
                n_bbq += 1

    # Top up BBQ examples so E10 gets n_samples (it only runs on BBQ)
    if n_bbq < n_samples:
        extra_needed = n_samples - n_bbq
        cfg = BENCHMARK_CONFIGS["bbq"]
        try:
            bm = get_benchmark("bbq")
            # Use different seed to avoid duplicates with the main pool
            extra_examples = bm.load_examples(
                n_samples=n_bbq + extra_needed, random_seed=seed + 100
            )
            # Skip ones we already have (by index offset)
            for ex in extra_examples[n_bbq:]:
                pert_id = rng.choice(cfg["perturbations"])
                test_set.append({
                    "global_idx": global_idx,
                    "benchmark": "bbq",
                    "domain": cfg["domain"],
                    "perturbation_id": pert_id,
                    "example": ex,
                })
                global_idx += 1
        except Exception as e:
            logger.warning(f"  Could not top up BBQ for E10: {e}")

    rng.shuffle(test_set)

    # Reassign global indices after shuffle
    for i, item in enumerate(test_set):
        item["global_idx"] = i

    return test_set


def compute_self_modeling_score(output_dir: Path, model_short: str, strict: bool = False) -> dict:
    """Compute unified self-modeling score by aggregating all group summaries.

    Normalizes all metrics to higher=better [0,1] and averages for overall score.

    Args:
        strict: If True, use strict metrics (parse failures = 0 for accuracy, 1.0 for MSE).
                If False (default), use lenient metrics (skip unparseable responses).

    Returns dict with per-eval scores and overall score.
    """
    import numpy as np

    all_metrics: dict[int, list[dict]] = {}  # eval_id -> list of metrics dicts

    # Collect summaries — they're nested: group_dir/run_dir/model_short/summary.json
    for summary_path in sorted(output_dir.rglob(f"{model_short}/summary.json")):
        with open(summary_path) as f:
            summary = json.load(f)
        for eid_str, metrics in summary.get("evals", {}).items():
            eid = int(eid_str)
            if "error" not in metrics:
                # In strict mode, use the "strict" sub-dict if available
                m = metrics.get("strict", metrics) if strict else metrics
                all_metrics.setdefault(eid, []).append(m)

    # Normalize each eval to higher=better [0,1]
    eval_scores = {}
    for eid, metrics_list in sorted(all_metrics.items()):
        if eid in (1,):
            # E1: accuracy — already [0,1], higher=better
            vals = [m.get("accuracy") for m in metrics_list if m.get("accuracy") is not None]
            eval_scores[eid] = {"name": "Flip Prediction", "score": float(np.mean(vals)) if vals else None, "metric": "accuracy", "n": len(vals)}
        elif eid == 2:
            # E2: mean_similarity — already [0,1], higher=better
            vals = [m.get("mean_similarity") for m in metrics_list if m.get("mean_similarity") is not None]
            eval_scores[eid] = {"name": "Output Prediction", "score": float(np.mean(vals)) if vals else None, "metric": "similarity", "n": len(vals)}
        elif eid == 3:
            # E3: mse — [0,1], lower=better → invert
            vals = [m.get("mse") for m in metrics_list if m.get("mse") is not None]
            eval_scores[eid] = {"name": "Flip Probability", "score": float(1.0 - np.mean(vals)) if vals else None, "metric": "1-mse", "n": len(vals)}
        elif eid == 4:
            # E4: mse — same
            vals = [m.get("mse") for m in metrics_list if m.get("mse") is not None]
            eval_scores[eid] = {"name": "Correctness Probability", "score": float(1.0 - np.mean(vals)) if vals else None, "metric": "1-mse", "n": len(vals)}
        elif eid == 5:
            # E5: mse — same
            vals = [m.get("mse") for m in metrics_list if m.get("mse") is not None]
            eval_scores[eid] = {"name": "Confidence After Pert", "score": float(1.0 - np.mean(vals)) if vals else None, "metric": "1-mse", "n": len(vals)}
        elif eid == 6:
            # E6: accuracy — already [0,1]
            vals = [m.get("accuracy") for m in metrics_list if m.get("accuracy") is not None]
            eval_scores[eid] = {"name": "Perturbation Ranking", "score": float(np.mean(vals)) if vals else None, "metric": "accuracy", "n": len(vals)}
        elif eid == 7:
            # E7: accuracy — already [0,1]
            vals = [m.get("accuracy") for m in metrics_list if m.get("accuracy") is not None]
            eval_scores[eid] = {"name": "Component Ablation", "score": float(np.mean(vals)) if vals else None, "metric": "accuracy", "n": len(vals)}
        elif eid == 8:
            # E8: flip_accuracy — already [0,1]
            vals = [m.get("flip_accuracy") for m in metrics_list if m.get("flip_accuracy") is not None]
            eval_scores[eid] = {"name": "Minimal Edit", "score": float(np.mean(vals)) if vals else None, "metric": "flip_accuracy", "n": len(vals)}
        elif eid == 9:
            # E9: mse — same
            vals = [m.get("mse") for m in metrics_list if m.get("mse") is not None]
            eval_scores[eid] = {"name": "Feature Presence", "score": float(1.0 - np.mean(vals)) if vals else None, "metric": "1-mse", "n": len(vals)}
        elif eid == 10:
            # E10: mixed margin/second — average mse_margin (inverted) + accuracy_second
            mse_vals = [m.get("mse_margin") for m in metrics_list if m.get("mse_margin") is not None]
            acc_vals = [m.get("accuracy_second") for m in metrics_list if m.get("accuracy_second") is not None]
            parts = []
            if mse_vals:
                parts.append(1.0 - float(np.mean(mse_vals)))
            if acc_vals:
                parts.append(float(np.mean(acc_vals)))
            eval_scores[eid] = {"name": "Margin & Second", "score": float(np.mean(parts)) if parts else None, "metric": "combined", "n": len(mse_vals) + len(acc_vals)}

    # Overall score
    valid_scores = [v["score"] for v in eval_scores.values() if v["score"] is not None]
    overall = float(np.mean(valid_scores)) if valid_scores else None

    return {"eval_scores": eval_scores, "overall": overall}


def compute_baselines(output_dir: Path, model_short: str) -> dict[str, float]:
    """Compute per-eval baselines from GT distributions in the CSV files.

    Baselines represent the expected score from a trivial predictor:
    - Accuracy evals (E1, E6, E7): majority class / random chance
    - MSE evals (E3, E4, E5, E9): 1-MSE of predicting the GT mean
    - E2 (similarity): 0 (random text)
    - E8 (flip accuracy): 0 (random edits don't flip)
    - E10 (combined): 0.5

    Returns dict mapping eval key (e.g. "E1") to baseline score.
    """
    import csv
    import numpy as np

    gt_vals: dict[str, list[float]] = {}
    gt_labels: dict[str, list[str]] = {}  # for MCQ majority class

    # Continuous GT values (for predict-mean baselines)
    csv_map = {
        "E1": ("eval01_flip_prediction.csv", "gt_flipped", lambda v: float(v == "True")),
        "E3": ("eval03_flip_probability.csv", "gt_flip_rate", float),
        "E4": ("eval04_correctness_probability.csv", "empirical_accuracy", float),
        "E5": ("eval05_confidence_calibration.csv", "gt_mean_confidence", float),
        "E9": ("eval09_feature_presence.csv", "empirical_feature_rate", float),
    }
    # MCQ GT labels (for majority class baselines)
    label_map = {
        "E6": ("eval06_perturbation_ranking.csv", "gt_letter"),
        "E7": ("eval07_prompt_component_ablation.csv", "gt_letter"),
        "E10_second": ("eval10_margin_and_second.csv", "gt_second"),
    }
    # E10 margin continuous values
    margin_map = {
        "E10_margin": ("eval10_margin_and_second.csv", "gt_margin"),
    }

    for eval_csv_path in sorted(output_dir.rglob(f"{model_short}/*.csv")):
        fname = eval_csv_path.name
        try:
            rows = list(csv.DictReader(open(eval_csv_path)))
        except Exception:
            continue

        # Continuous values
        for key, (target_fname, col, parser) in csv_map.items():
            if fname == target_fname:
                gt_vals.setdefault(key, [])
                for row in rows:
                    raw = row.get(col, "")
                    if raw:
                        try:
                            gt_vals[key].append(parser(raw))
                        except (ValueError, TypeError):
                            pass

        # MCQ labels
        for key, (target_fname, col) in label_map.items():
            if fname == target_fname:
                gt_labels.setdefault(key, [])
                for row in rows:
                    raw = row.get(col, "")
                    # For E10, only count rows with the "second" template
                    if key == "E10_second" and row.get("template") != "second":
                        continue
                    if raw:
                        gt_labels[key].append(raw)

        # E10 margin continuous
        for key, (target_fname, col) in margin_map.items():
            if fname == target_fname:
                gt_vals.setdefault(key, [])
                for row in rows:
                    if row.get("template") != "margin":
                        continue
                    raw = row.get(col, "")
                    if raw:
                        try:
                            gt_vals[key].append(float(raw))
                        except (ValueError, TypeError):
                            pass

    baselines: dict[str, float] = {}

    # E1: majority class accuracy (binary: flipped or not)
    if gt_vals.get("E1"):
        arr = np.array(gt_vals["E1"])
        baselines["E1"] = float(max(arr.mean(), 1 - arr.mean()))
    else:
        baselines["E1"] = 0.5

    # E2: random text similarity
    baselines["E2"] = 0.0

    # E3, E4, E5, E9: 1 - variance (= 1-MSE of predicting GT mean)
    for key in ("E3", "E4", "E5", "E9"):
        if gt_vals.get(key):
            arr = np.array(gt_vals[key])
            baselines[key] = float(1.0 - np.var(arr))
        else:
            baselines[key] = 0.5

    # E6: majority class accuracy (MCQ — which perturbation is strongest)
    if gt_labels.get("E6"):
        from collections import Counter
        counts = Counter(gt_labels["E6"])
        baselines["E6"] = float(counts.most_common(1)[0][1] / len(gt_labels["E6"]))
    else:
        baselines["E6"] = 1 / 3

    # E7: majority class accuracy (MCQ — which component matters most)
    if gt_labels.get("E7"):
        from collections import Counter
        counts = Counter(gt_labels["E7"])
        baselines["E7"] = float(counts.most_common(1)[0][1] / len(gt_labels["E7"]))
    else:
        baselines["E7"] = 0.5

    # E8: random edit won't flip
    baselines["E8"] = 0.0

    # E10: combined — majority class for second + predict-mean for margin
    e10_parts = []
    if gt_vals.get("E10_margin"):
        arr = np.array(gt_vals["E10_margin"])
        e10_parts.append(float(1.0 - np.var(arr)))  # 1-MSE of predict-mean
    if gt_labels.get("E10_second"):
        from collections import Counter
        counts = Counter(gt_labels["E10_second"])
        e10_parts.append(float(counts.most_common(1)[0][1] / len(gt_labels["E10_second"])))
    baselines["E10"] = float(np.mean(e10_parts)) if e10_parts else 0.5

    return baselines


def compute_skill_scores(
    eval_scores: dict, baselines: dict[str, float]
) -> tuple[dict[str, dict], float | None]:
    """Compute skill scores: score - baseline (unnormalized).

    Simple difference bounded in [-1, 1] since both score and baseline
    are in [0, 1]. Previous formula (score - base) / (1 - base) was
    unbounded and blew up when baseline ≈ 1 (e.g., E5).

    Returns (per_eval_skills, overall_skill).
    """
    import numpy as np

    skills: dict[str, dict] = {}
    for eid, info in eval_scores.items():
        key = f"E{eid}" if isinstance(eid, int) else eid
        raw = info.get("score")
        base = baselines.get(key, 0.0)
        if raw is not None:
            skill = raw - base
            skills[key] = {"score": raw, "baseline": base, "skill": float(skill)}
        else:
            skills[key] = {"score": raw, "baseline": base, "skill": None}

    valid_skills = [v["skill"] for v in skills.values() if v["skill"] is not None]
    overall = float(np.mean(valid_skills)) if valid_skills else None
    return skills, overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run mixed self-modeling evals")
    parser.add_argument("--models", required=True, nargs="+")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-resample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--output-dir", default="outputs/results/mixed_self_modeling")
    parser.add_argument("--eval-ids", type=int, nargs="+", default=None)
    parser.add_argument("--resample-temp", type=float, default=0.0)
    parser.add_argument("--phase2-temp", type=float, default=0.0,
                        help="Temperature for Phase 2 self-modeling queries (default: 0.0)")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent API calls (default: 20)")
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable extended thinking for Haiku 4.5+ (budget_tokens via --budget-tokens)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable thinking even for models that auto-detect as thinking (e.g., Qwen3). "
             "Sends chat_template_kwargs.enable_thinking=False to vLLM.",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Skip the self-modeling system prompt (ablation test).",
    )
    parser.add_argument(
        "--budget-tokens", type=int, default=1024,
        help="Thinking budget tokens (default: 1024)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    logger.info(f"\n{_GREEN}[INFO]{_RESET} Building mixed test set ({args.n_samples} samples, seed={args.seed})...")
    test_set = build_mixed_test_set(args.n_samples, args.seed)

    # Summary
    from collections import Counter
    bm_counts = Counter(item["benchmark"] for item in test_set)
    pert_counts = Counter(item["perturbation_id"] for item in test_set)

    logger.info(f"  Benchmarks: {dict(bm_counts)}")
    logger.info(f"  Perturbations: {dict(pert_counts)}")
    logger.info(f"  Total: {len(test_set)} examples")

    # Save test set manifest
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"mixed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for item in test_set:
        manifest.append({
            "global_idx": item["global_idx"],
            "benchmark": item["benchmark"],
            "domain": item["domain"],
            "perturbation_id": item["perturbation_id"],
            "example_idx": item["example"].idx,
            "question_preview": item["example"].question[:100],
        })

    with open(output_dir / "test_set_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  Manifest saved: {output_dir / 'test_set_manifest.json'}")

    # Group by (benchmark, perturbation) for efficient batching
    from collections import defaultdict
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for item in test_set:
        groups[(item["benchmark"], item["perturbation_id"])].append(item)

    logger.info(f"  Groups: {len(groups)} unique (benchmark, perturbation) pairs")

    # For each group, run self-modeling evals
    from prompt_attribution.eval.self_modeling.config import SelfModelingConfig
    from prompt_attribution.eval.self_modeling.runner import SelfModelingRunner

    for (bm_name, pert_id), items in sorted(groups.items()):
        n = len(items)
        logger.info(f"\n{_CYAN}[STEP]{_RESET} {bm_name}/{pert_id} ({n} examples)")

        # Determine which evals to run for this group
        eval_ids = args.eval_ids
        if eval_ids is None:
            eval_ids = list(range(1, 10))  # E1-E9 for all
            if bm_name == "bbq":
                eval_ids.append(10)  # E10 only for BBQ

        # Pick 3 perturbations from the same domain for eval6 ranking
        all_perts = BENCHMARK_CONFIGS[bm_name]["perturbations"]
        # Pick 3 for eval6 (include current + 2 others)
        eval6_perts = [pert_id]
        for p in all_perts:
            if p != pert_id and len(eval6_perts) < 3:
                eval6_perts.append(p)

        config = SelfModelingConfig(
            benchmark=bm_name,
            perturbation_id=pert_id,
            models=args.models,
            n_samples=n,
            n_resample=args.n_resample,
            resample_temperature=args.resample_temp,
            phase2_temperature=args.phase2_temp,
            random_seed=args.seed,
            eval_ids=eval_ids,
            eval6_perturbation_ids=eval6_perts if len(eval6_perts) == 3 else None,
            max_concurrent=args.max_concurrent,
            vllm_url=args.vllm_url,
            thinking=args.thinking,
            no_thinking=args.no_thinking,
            no_system_prompt=args.no_system_prompt,
            thinking_budget_tokens=args.budget_tokens,
            output_dir=str(output_dir / f"{bm_name}_{pert_id}"),
        )

        runner = SelfModelingRunner(config)
        try:
            await runner.run()
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Save CLI command for reproducibility
    with open(output_dir / "command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Aggregated scores per model
    from prompt_attribution.eval.self_modeling.runner import _short_model_name
    for model_id in args.models:
        model_short = _short_model_name(model_id)

        # Lenient scores (skip NA)
        scores = compute_self_modeling_score(output_dir, model_short, strict=False)
        overall = scores['overall']
        overall_str = f"{overall:.3f}" if overall is not None else "N/A"
        logger.info(f"{_GREEN}[INFO]{_RESET} Overall self-modeling score ({model_id}): {overall_str}")
        for eid, info in sorted(scores["eval_scores"].items()):
            s = info["score"]
            s_str = f"{s:.3f}" if s is not None else "N/A"
            logger.info(f"  E{eid} {info['name']}: {s_str}")

        # Strict scores (NA=0)
        strict_scores = compute_self_modeling_score(output_dir, model_short, strict=True)
        strict_overall = strict_scores['overall']
        strict_str = f"{strict_overall:.3f}" if strict_overall is not None else "N/A"
        logger.info(f"\n{_GREEN}[INFO]{_RESET} Strict score (NA=0) ({model_id}): {strict_str}")
        for eid, info in sorted(strict_scores["eval_scores"].items()):
            s = info["score"]
            s_str = f"{s:.3f}" if s is not None else "N/A"
            logger.info(f"  E{eid} {info['name']}: {s_str}")

        # Save per-model aggregated scores (lenient + strict) + config
        # Count total examples per eval across all groups
        example_counts = {}
        for summary_path in sorted(output_dir.rglob(f"{model_short}/summary.json")):
            with open(summary_path) as sf:
                summary = json.load(sf)
            for eid_str, m in summary.get("evals", {}).items():
                if "error" not in m:
                    example_counts.setdefault(int(eid_str), 0)
                    example_counts[int(eid_str)] += m.get("n_total", 0)

        # Compute baselines and skill scores
        baselines = compute_baselines(output_dir, model_short)
        lenient_skills, lenient_skill_overall = compute_skill_scores(
            scores.get("eval_scores", {}), baselines,
        )
        strict_skills, strict_skill_overall = compute_skill_scores(
            strict_scores.get("eval_scores", {}), baselines,
        )

        # Log skill scores
        _MAGENTA = "\033[95m"
        ls_str = f"{lenient_skill_overall:.3f}" if lenient_skill_overall is not None else "N/A"
        ss_str = f"{strict_skill_overall:.3f}" if strict_skill_overall is not None else "N/A"
        logger.info(f"\n{_MAGENTA}[EVAL]{_RESET} Skill scores ({model_id}):")
        logger.info(f"  Lenient skill: {ls_str}   Strict skill: {ss_str}")
        for key in sorted(lenient_skills.keys(), key=lambda k: int(k[1:])):
            li = lenient_skills[key]
            si = strict_skills.get(key, {})
            l_sk = f"{li['skill']:.3f}" if li.get("skill") is not None else "N/A"
            s_sk = f"{si['skill']:.3f}" if si.get("skill") is not None else "N/A"
            base = li.get("baseline", 0)
            logger.info(f"  {key}: lenient={l_sk}  strict={s_sk}  (baseline={base:.3f})")

        agg_entry = {
            "model_id": model_id,
            "model_short": model_short,
            "config": {
                "n_samples": args.n_samples,
                "n_resample": args.n_resample,
                "seed": args.seed,
                "resample_temp": args.resample_temp,
                "phase2_temp": args.phase2_temp,
                "max_concurrent": args.max_concurrent,
                "eval_ids": args.eval_ids,
                "vllm_url": getattr(args, "vllm_url", None),
            },
            "lenient_overall": scores["overall"],
            "strict_overall": strict_scores["overall"],
            "lenient_skill_overall": lenient_skill_overall,
            "strict_skill_overall": strict_skill_overall,
            "baselines": baselines,
            "lenient_per_eval": {
                f"E{eid}": {
                    "score": info["score"],
                    "metric": info["metric"],
                    "n_groups": info["n"],
                    "n_examples": example_counts.get(eid, 0),
                    "baseline": baselines.get(f"E{eid}", 0.0),
                    "skill": lenient_skills.get(f"E{eid}", {}).get("skill"),
                }
                for eid, info in scores.get("eval_scores", {}).items()
            },
            "strict_per_eval": {
                f"E{eid}": {
                    "score": info["score"],
                    "metric": info.get("metric", ""),
                    "n_groups": info.get("n", 0),
                    "n_examples": example_counts.get(eid, 0),
                    "baseline": baselines.get(f"E{eid}", 0.0),
                    "skill": strict_skills.get(f"E{eid}", {}).get("skill"),
                }
                for eid, info in strict_scores.get("eval_scores", {}).items()
            },
        }
        agg_path = output_dir / f"scores_{model_short}.json"
        with open(agg_path, "w") as f:
            json.dump(agg_entry, f, indent=2)
        logger.info(f"{_GREEN}[INFO]{_RESET} Aggregated scores: {agg_path}")

    logger.info(f"\n{_GREEN}[INFO]{_RESET} All groups done. Results at: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
