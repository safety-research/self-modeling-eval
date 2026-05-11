"""
Configuration module - dataclasses and YAML config loading.

Structure:
- experiment_config.py: PerturbationConfig, perturbation/domain/registry loaders,
  get_provider (model id -> provider routing)
- model_format.py: ModelFormat (thinking-mode auto-detection)
- registry.yaml: maps benchmarks to domains
- perturbations/: per-domain YAML perturbation configs
"""

from .experiment_config import (
    PerturbationConfig,
    load_registry,
    load_perturbation,
    load_domain_config,
    get_domain_for_benchmark,
    get_provider,
)

__all__ = [
    "PerturbationConfig",
    "load_registry",
    "load_perturbation",
    "load_domain_config",
    "get_domain_for_benchmark",
    "get_provider",
]
