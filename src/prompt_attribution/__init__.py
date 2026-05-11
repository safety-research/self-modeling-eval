"""
Self-modeling eval — standalone mixed self-modeling eval suite (E1–E10).

Structure:
- eval/
  - self_modeling/: E1-E10 evals, runner, GT engine, parsers
  - benchmarks/: dataset loaders (GSM8K, HumanEval, BBQ, WildGuardTest)
  - domains/: domain-specific verifiers (math / coding / fairness / safety)
- shared/
  - config/: PerturbationConfig, registry.yaml, perturbations/*.yaml,
    model_format.py (thinking-mode auto-detection)
"""

__version__ = "0.2.0"
