"""
Module: prompt_attribution/domains/fairness/__init__.py

Fairness domain for bias/fairness testing with multiple-choice questions.
Works with BBQ (Bias Benchmark for QA) and similar MCQ benchmarks.
"""

from .domain import FairnessDomain
from .verifier import FairnessVerifier

__all__ = ["FairnessDomain", "FairnessVerifier"]
