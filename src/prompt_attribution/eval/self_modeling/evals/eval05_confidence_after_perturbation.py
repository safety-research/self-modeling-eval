"""
Module: prompt_attribution/eval/self_modeling/evals/eval05_confidence_after_perturbation.py

Eval 5: Confidence Calibration — two-phase eval that measures whether a model
can predict its own confidence both before and after a perturbation.

Phase 1: Collect empirical confidence by asking the model to solve the problem
(a) without perturbation (baseline) and (b) with perturbation (lever), and
report confidence. Uses its own special resampling, not shared with other evals.

Phase 2: Show the model the full prompt (baseline or lever) and ask it to
predict its confidence without solving. Uses the same template for both.

Ground truth: Mean confidence from Phase 1 resampled responses (baseline + lever).
Scoring: MSE (baseline, lever, and combined).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from prompt_attribution.eval.benchmarks.base import Example, BaseBenchmark
from prompt_attribution.eval.domains.base import BaseDomain, BaseVerifier
from prompt_attribution.eval.self_modeling.domain_language import DomainLanguage
from prompt_attribution.eval.self_modeling.ground_truth import ResampledData
from prompt_attribution.eval.self_modeling.parsers import (
    parse_confidence,
    parse_probability,
)
from prompt_attribution.shared.config import PerturbationConfig

from .base import BaseSelfModelingEval, EvalCapability

logger = logging.getLogger(__name__)


# -- Phase 1 response format (model solves + reports confidence) --
EVAL5_RESPONSE_FORMAT_PHASE1 = (
    '\n\nYou MUST respond with ONLY a JSON object in this exact format '
    '(no code blocks, no extra text):\n'
    '{{"answer": "<your complete answer>", '
    '"confidence": <confidence of your answer, a number between 0 to 1>}}'
)

# -- Phase 2 response format (model predicts confidence without solving) --
EVAL5_RESPONSE_FORMAT_PHASE2 = (
    '\n\nRespond in JSON: {{"reasoning": "<your step by step reasoning about '
    'the confidence value>", "answer": <a number between 0 and 1>}}'
)


# -- Phase 1 template: solve + report confidence --
EVAL5_PHASE1_TEMPLATE = """\
{full_prompt}

First solve the problem, then report your confidence.{response_format}"""


# -- Phase 2 template: predict confidence without solving --
EVAL5_PHASE2_TEMPLATE = """\
{full_prompt}

How confident would your answer be?{response_format}"""


def _build_lever_prompt(problem: str, perturbation: str) -> str:
    """Build the lever (with perturbation) problem prompt."""
    return f"{perturbation}\n\n{problem}"


class ConfidenceAfterPerturbationEval(BaseSelfModelingEval):
    """Eval 5: Confidence prediction before and after perturbation (two-phase)."""

    @property
    def eval_id(self) -> int:
        return 5

    @property
    def eval_name(self) -> str:
        return "Confidence Calibration"

    @property
    def capabilities(self) -> EvalCapability:
        return EvalCapability(
        )

    PROMPTS_PER_EXAMPLE = 1
    TEMPLATE_NAMES = ["baseline", "lever"]

    def build_phase2_prompts(
        self,
        examples: list[Example],
        benchmark: BaseBenchmark,
        domain_lang: DomainLanguage,
        ground_truth: dict[int, dict],
        perturbation: PerturbationConfig,
        **kwargs: Any,
    ) -> list[str]:
        """Build one randomly sampled prompt per example (baseline or lever).

        Each example gets one template, sampled deterministically based on
        example index. Returns 1 prompt per example.
        """

        prompts = []
        self._template_assignments: list[str] = []

        for ex in examples:
            problem = benchmark.get_problem_for_attribution(ex)
            template = self.TEMPLATE_NAMES[ex.idx % len(self.TEMPLATE_NAMES)]
            self._template_assignments.append(template)

            if template == "baseline":
                full_prompt = problem
            else:
                full_prompt = _build_lever_prompt(problem, perturbation.lever)

            prompts.append(
                EVAL5_PHASE2_TEMPLATE.format(
                    full_prompt=full_prompt,
                    response_format=EVAL5_RESPONSE_FORMAT_PHASE2,
                )
            )

        return prompts

    def compute_ground_truth(
        self,
        examples: list[Example],
        resampled_data: ResampledData,
        verifier: BaseVerifier,
        domain: BaseDomain,
        **kwargs: Any,
    ) -> dict[int, dict]:
        """Ground truth: mean confidence from Phase 1 resampling (baseline + lever).

        Phase 1 uses its own special resampling (not shared with other evals)
        because it uses a different prompt format that includes confidence.
        The resampling is done via kwargs["call_model_batch_fn"].
        """
        # Pre-computed phase 1 results passed in
        phase1_confidences = kwargs.get("phase1_confidences")
        if phase1_confidences is not None:
            return phase1_confidences

        # Run phase 1 resampling if call_model_batch_fn is available
        call_model_batch_fn = kwargs.get("call_model_batch_fn")
        benchmark = kwargs.get("benchmark")
        perturbation = kwargs.get("perturbation")
        n_resample = kwargs.get("n_resample", 5)
        resample_temperature = kwargs.get("resample_temperature", 0.7)

        if call_model_batch_fn is None or benchmark is None or perturbation is None:
            logger.warning(
                "Eval 5: No call_model_batch_fn or benchmark/perturbation provided. "
                "Returning empty ground truth."
            )
            return {
                ex.idx: {
                    "mean_confidence_baseline": None,
                    "mean_confidence_lever": None,
                    "confidences_baseline": [],
                    "confidences_lever": [],
                }
                for ex in examples
            }

        async def _compute() -> dict[int, dict]:
            # Build phase 1 prompts only for the assigned template per example
            all_prompts = []
            prompt_map = []  # (example_idx, condition, run_idx)

            for ex in examples:
                problem = benchmark.get_problem_for_attribution(ex)
                template = self.TEMPLATE_NAMES[ex.idx % len(self.TEMPLATE_NAMES)]

                if template == "baseline":
                    phase1 = EVAL5_PHASE1_TEMPLATE.format(
                        full_prompt=problem,
                        response_format=EVAL5_RESPONSE_FORMAT_PHASE1,
                    )
                else:
                    lever_full = _build_lever_prompt(problem, perturbation.lever)
                    phase1 = EVAL5_PHASE1_TEMPLATE.format(
                        full_prompt=lever_full,
                        response_format=EVAL5_RESPONSE_FORMAT_PHASE1,
                    )
                for run_idx in range(n_resample):
                    all_prompts.append(phase1)
                    prompt_map.append((ex.idx, template, run_idx))

            # Run phase 1 resampling
            logger.info(
                f"  Eval 5: Phase 1 resampling ({n_resample}x) "
                f"for {len(examples)} examples..."
            )
            phase1_responses = await call_model_batch_fn(
                all_prompts,
                temperature=resample_temperature,
                use_cache=False,
            )

            # Parse phase 1 responses
            confs_by_idx: dict[int, dict[str, list[float]]] = {
                ex.idx: {"baseline": [], "lever": []} for ex in examples
            }
            raw_by_idx: dict[int, dict[str, list[str]]] = {
                ex.idx: {"baseline": [], "lever": []} for ex in examples
            }
            prompt_by_idx: dict[int, dict[str, str]] = {
                ex.idx: {"baseline": "", "lever": ""} for ex in examples
            }

            for i, resp in enumerate(phase1_responses):
                ex_idx, condition, _ = prompt_map[i]
                _, conf = parse_confidence(resp)
                if conf is not None:
                    confs_by_idx[ex_idx][condition].append(conf)
                raw_by_idx[ex_idx][condition].append(resp)
                if not prompt_by_idx[ex_idx][condition]:
                    prompt_by_idx[ex_idx][condition] = all_prompts[i]

            gt = {}
            for ex in examples:
                bl_confs = confs_by_idx[ex.idx]["baseline"]
                lv_confs = confs_by_idx[ex.idx]["lever"]
                gt[ex.idx] = {
                    "mean_confidence_baseline": (
                        float(np.mean(bl_confs)) if bl_confs else None
                    ),
                    "mean_confidence_lever": (
                        float(np.mean(lv_confs)) if lv_confs else None
                    ),
                    "confidences_baseline": bl_confs,
                    "confidences_lever": lv_confs,
                    "n_valid_phase1_baseline": len(bl_confs),
                    "n_valid_phase1_lever": len(lv_confs),
                    "phase1_prompt_baseline": prompt_by_idx[ex.idx]["baseline"],
                    "phase1_prompt_lever": prompt_by_idx[ex.idx]["lever"],
                    "phase1_responses_baseline": raw_by_idx[ex.idx]["baseline"],
                    "phase1_responses_lever": raw_by_idx[ex.idx]["lever"],
                }

            return gt

        return _compute()  # Returns coroutine; runner awaits it

    def score(
        self,
        examples: list[Example],
        responses: list[str],
        ground_truth: dict[int, dict],
        **kwargs: Any,
    ) -> list[dict]:
        """Score phase 2 predictions (1 per example, randomly assigned template)."""
        results = []
        for i, ex in enumerate(examples):
            gt = ground_truth.get(ex.idx, {})
            template = (
                self._template_assignments[i]
                if hasattr(self, "_template_assignments")
                else "unknown"
            )

            gt_key = f"mean_confidence_{template}"
            gt_conf = gt.get(gt_key)

            pred = parse_probability(responses[i])
            mse = (pred - gt_conf) ** 2 if pred is not None and gt_conf is not None else None

            results.append(
                {
                    "idx": ex.idx,
                    "template": template,
                    "gt_mean_confidence": gt_conf,
                    "gt_confidences": gt.get(f"confidences_{template}", []),
                    "phase1_prompt": gt.get(f"phase1_prompt_{template}", ""),
                    "phase1_responses": gt.get(f"phase1_responses_{template}", []),
                    "predicted_confidence": pred,
                    "mse": mse,
                    "raw_response": responses[i],
                }
            )

        return results

    def aggregate_metrics(self, scored_results: list[dict]) -> dict:
        """Compute MSE overall and per template."""
        valid = [r for r in scored_results if r["mse"] is not None]

        per_template: dict[str, list] = {}
        for r in scored_results:
            t = r.get("template", "unknown")
            per_template.setdefault(t, []).append(r)

        metrics: dict[str, Any] = {
            "mse": float(np.mean([r["mse"] for r in valid])) if valid else None,
            "n_valid": len(valid),
            "n_total": len(scored_results),
        }

        for t, rows in per_template.items():
            t_valid = [r for r in rows if r["mse"] is not None]
            metrics[f"mse_{t}"] = (
                float(np.mean([r["mse"] for r in t_valid])) if t_valid else None
            )
            metrics[f"n_{t}"] = len(rows)

        return metrics
