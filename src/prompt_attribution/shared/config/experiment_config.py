"""
Module: prompt_attribution/shared/config/experiment_config.py

Structure:
- PerturbationConfig: dataclass loaded from perturbations/<domain>.yaml
- get_provider(): map model id → safetytooling provider name
- load_registry(): parse registry.yaml (benchmark → domain)
- get_domain_for_benchmark(): registry lookup
- load_perturbation(): build a PerturbationConfig from YAML
- load_domain_config(): per-domain extras (e.g. attribution_question)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# Provider routing: model-id prefixes that get routed to Together AI.
TOGETHER_PREFIXES = (
    "meta-llama/",
    "Qwen/",
    "deepseek-ai/",
    "mistralai/",
    "google/",
    "databricks/",
    "NousResearch/",
    "togethercomputer/",
)


@dataclass
class PerturbationConfig:
    """Configuration for a perturbation (loaded from YAML)."""
    perturbation_id: str
    description: str
    baseline: str
    lever: str
    # E9 feature: baseline behavioral feature (benchmark-level, e.g. response length)
    feature_description: Optional[str] = None
    target_features: Optional[list[str]] = None
    feature_target_value: Optional[str] = None
    # E1/E3 flip feature: what the perturbation changes (perturbation-driven)
    flip_feature_description: Optional[str] = None
    flip_target_features: Optional[list[str]] = None
    is_control: bool = False


def get_provider(model_id: str) -> str:
    """Determine provider from model ID.

    Returns one of: "anthropic", "openai", "gemini", "openrouter", "together".
    Raises ValueError if no rule matches.
    """
    if model_id.startswith("claude"):
        return "anthropic"
    elif model_id.startswith("gpt") or model_id.startswith(("o1", "o3", "o4")):
        return "openai"
    elif model_id.startswith("gemini"):
        return "gemini"
    elif model_id.startswith("openrouter/"):
        return "openrouter"
    elif model_id.startswith(("moonshotai/", "x-ai/")):
        return "openrouter"
    elif any(model_id.startswith(prefix) for prefix in TOGETHER_PREFIXES):
        return "together"
    else:
        raise ValueError(f"Unknown provider for model: {model_id}")


def _get_config_dir() -> Path:
    return Path(__file__).parent


def load_registry() -> dict:
    """Load benchmark-domain registry from registry.yaml."""
    with open(_get_config_dir() / "registry.yaml") as f:
        data = yaml.safe_load(f)
    return data.get("benchmarks", {})


def get_domain_for_benchmark(benchmark: str) -> str:
    """Get the domain for a benchmark from registry."""
    registry = load_registry()
    if benchmark not in registry:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[benchmark]["domain"]


def load_perturbation(domain: str, perturbation_id: str) -> PerturbationConfig:
    """Build a PerturbationConfig from perturbations/<domain>.yaml."""
    yaml_path = _get_config_dir() / "perturbations" / f"{domain}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"No perturbation config for domain '{domain}'")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    perturbations = data.get("perturbations", {})
    if perturbation_id not in perturbations:
        raise ValueError(
            f"Unknown perturbation '{perturbation_id}' for domain '{domain}'. "
            f"Available: {list(perturbations.keys())}"
        )

    config = perturbations[perturbation_id]
    return PerturbationConfig(
        perturbation_id=perturbation_id,
        description=config.get("description", ""),
        baseline=config.get("baseline", ""),
        lever=config.get("lever", ""),
        feature_description=config.get("feature_description"),
        target_features=config.get("target_features"),
        feature_target_value=config.get("feature_target_value"),
        flip_feature_description=config.get("flip_feature_description"),
        flip_target_features=config.get("flip_target_features"),
        is_control=config.get("is_control", False),
    )


def load_domain_config(domain: str) -> dict:
    """Load domain-level extras (e.g. attribution_question)."""
    yaml_path = _get_config_dir() / "perturbations" / f"{domain}.yaml"
    defaults = {"attribution_question": "Would your answer be different?"}
    if not yaml_path.exists():
        return defaults
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    config = data.get("domain_config", {})
    return {**defaults, **config}
