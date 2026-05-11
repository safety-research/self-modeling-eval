"""
Module: prompt_attribution/eval/domains/safety/refusal_classifier.py

`RefusalResult` is the dataclass used by `ground_truth.py` and
`eval/self_modeling/evals/base.py` to store judge-LLM refusal
classifications. The judge is called directly from
`classify_safety_responses_batch`.
"""

from dataclasses import dataclass


@dataclass
class RefusalResult:
    """Result of refusal classification."""

    is_refusal: bool
    raw_classification: str  # Raw judge output (e.g., "Refusal" or "Non-Refusal")
    query: str
    response_snippet: str  # First N chars of response
