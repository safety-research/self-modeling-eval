"""
Benchmark data loaders.

Only the four benchmarks sampled by the mixed self-modeling eval are kept:

- base.py: BaseBenchmark abstract class + Example dataclass
- gsm8k.py: GSM8KBenchmark
- humaneval.py: HumanEvalBenchmark
- bbq.py: BBQBenchmark
- wildguardtest.py: WildGuardTestBenchmark
"""

from .base import BaseBenchmark, Example
from .gsm8k import GSM8KBenchmark
from .humaneval import HumanEvalBenchmark
from .wildguardtest import WildGuardTestBenchmark
from .bbq import BBQBenchmark


BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    "gsm8k": GSM8KBenchmark,
    "humaneval": HumanEvalBenchmark,
    "wildguardtest": WildGuardTestBenchmark,
    "bbq": BBQBenchmark,
}


def get_benchmark(name: str) -> BaseBenchmark:
    """Get a benchmark instance by name."""
    benchmark_cls = BENCHMARKS.get(name)
    if not benchmark_cls:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return benchmark_cls()


__all__ = [
    "BaseBenchmark",
    "Example",
    "GSM8KBenchmark",
    "HumanEvalBenchmark",
    "WildGuardTestBenchmark",
    "BBQBenchmark",
    "BENCHMARKS",
    "get_benchmark",
]
