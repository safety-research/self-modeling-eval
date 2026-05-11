"""
Module: prompt_attribution/eval/self_modeling/

Mixed Self-Modeling Evals — tests whether models can reason about their own behavior.

10 eval types across 4 benchmarks (GSM8K, HumanEval, BBQ, WildGuardTest)
and four perturbation domains (math / coding / safety / fairness).

Structure:
- config.py: SelfModelingConfig dataclass
- domain_language.py: DomainLanguage abstraction for cross-domain prompt templates
- parsers.py: Response parsers (JSON, yes/no, probability, letter)
- ground_truth.py: ResamplingEngine for shared GT computation
- runner.py: SelfModelingRunner orchestration
- evals/: Per-eval implementations (eval01 through eval10)
"""

from .config import SelfModelingConfig
from .runner import SelfModelingRunner

__all__ = ["SelfModelingConfig", "SelfModelingRunner"]
