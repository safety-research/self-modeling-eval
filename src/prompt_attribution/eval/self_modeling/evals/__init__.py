"""
Module: prompt_attribution/eval/self_modeling/evals/

Registry of all self-modeling evals (1-10).

Structure:
- EVAL_REGISTRY: dict mapping eval_id -> eval class
- get_eval(): Factory function to instantiate an eval by ID
- get_compatible_evals(): Filter evals by benchmark/domain compatibility
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseSelfModelingEval
    from prompt_attribution.eval.benchmarks.base import BaseBenchmark
    from prompt_attribution.eval.domains.base import BaseDomain
    from prompt_attribution.shared.config import PerturbationConfig


def _build_registry() -> dict[int, type]:
    """Lazily build the eval registry to avoid circular imports."""
    from .eval01_flip_prediction import FlipPredictionEval
    from .eval02_output_prediction import OutputPredictionEval
    from .eval03_flip_probability import FlipProbabilityEval
    from .eval04_correctness_probability import CorrectnessProbabilityEval
    from .eval05_confidence_after_perturbation import ConfidenceAfterPerturbationEval
    from .eval06_perturbation_ranking import PerturbationRankingEval
    from .eval07_prompt_component_ablation import PromptComponentAblationEval
    from .eval08_propose_flip_instruction import ProposeFlipInstructionEval
    from .eval09_feature_presence import FeaturePresenceEval
    from .eval10_margin_and_second import MarginAndSecondEval

    return {
        1: FlipPredictionEval,
        2: OutputPredictionEval,
        3: FlipProbabilityEval,
        4: CorrectnessProbabilityEval,
        5: ConfidenceAfterPerturbationEval,
        6: PerturbationRankingEval,
        7: PromptComponentAblationEval,
        8: ProposeFlipInstructionEval,
        9: FeaturePresenceEval,
        10: MarginAndSecondEval,
    }


_REGISTRY: dict[int, type] | None = None


def _get_registry() -> dict[int, type]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_compatible_evals(
    eval_ids: list[int] | None,
    benchmark: "BaseBenchmark",
    domain: "BaseDomain",
    perturbation: "PerturbationConfig",
) -> list["BaseSelfModelingEval"]:
    """Get eval instances filtered by compatibility with benchmark/domain.

    Args:
        eval_ids: Specific eval IDs to include (None = all)
        benchmark: Benchmark instance
        domain: Domain instance
        perturbation: Perturbation config

    Returns:
        List of compatible, instantiated eval objects
    """
    registry = _get_registry()
    ids = eval_ids if eval_ids is not None else sorted(registry.keys())

    evals = []
    for eid in ids:
        if eid not in registry:
            continue
        eval_obj = registry[eid]()
        if eval_obj.is_compatible(benchmark, domain, perturbation):
            evals.append(eval_obj)

    return evals
