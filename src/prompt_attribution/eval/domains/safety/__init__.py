"""
Safety domain implementation.

Structure:
- domain.py: SafetyDomain
- verifier.py: SafetyVerifier
- refusal_classifier.py: RefusalResult (judge-LLM result dataclass)
"""

from .domain import SafetyDomain
from .verifier import SafetyVerifier
from .refusal_classifier import RefusalResult

__all__ = ["SafetyDomain", "SafetyVerifier", "RefusalResult"]
