"""
Domain-specific implementations.

Structure:
- base.py: BaseDomain abstract class
- math/: MathDomain, MathVerifier
- coding/: CodingDomain, CodeVerifier, AST features
- safety/: SafetyDomain, SafetyVerifier
- fairness/: FairnessDomain, FairnessVerifier
"""

from .base import BaseDomain, BaseVerifier
from .math import MathDomain, MathVerifier
from .coding import CodingDomain, CodeVerifier
from .safety import SafetyDomain, SafetyVerifier
from .fairness import FairnessDomain, FairnessVerifier

from prompt_attribution.shared.config import PerturbationConfig


# Domain registry - maps domain names to domain classes
DOMAINS = {
    "math": MathDomain,
    "coding": CodingDomain,
    "safety": SafetyDomain,
    "fairness": FairnessDomain,
}


def create_domain(domain_name: str, perturbation_config: PerturbationConfig) -> BaseDomain:
    """Create domain instance by name."""
    domain_cls = DOMAINS.get(domain_name)
    if not domain_cls:
        raise ValueError(
            f"Unknown domain '{domain_name}'. "
            f"Available: {list(DOMAINS.keys())}"
        )
    return domain_cls(perturbation_config)


__all__ = [
    "BaseDomain",
    "BaseVerifier",
    "MathDomain",
    "MathVerifier",
    "CodingDomain",
    "CodeVerifier",
    "SafetyDomain",
    "SafetyVerifier",
    "FairnessDomain",
    "FairnessVerifier",
    "DOMAINS",
    "create_domain",
]
