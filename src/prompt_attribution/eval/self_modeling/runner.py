"""
Module: prompt_attribution/eval/self_modeling/runner.py

Orchestrates self-modeling evals across models.

Pipeline per model:
1. Load examples from benchmark
2. Resample baseline + lever (shared across all evals)
3. For each selected eval: compute GT → build prompts → inference → score → aggregate
4. Write results (CSV, JSON, per-group summary)

Structure:
- SelfModelingRunner: Main orchestration class
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from prompt_attribution.eval.benchmarks.base import BaseBenchmark, Example
from prompt_attribution.eval.domains.base import BaseDomain, BaseVerifier
from prompt_attribution.eval.self_modeling.config import SelfModelingConfig
from prompt_attribution.eval.self_modeling.domain_language import (
    DomainLanguage,
    build_domain_language,
)
from prompt_attribution.eval.self_modeling.evals import get_compatible_evals
from prompt_attribution.eval.self_modeling.evals.base import BaseSelfModelingEval
from prompt_attribution.eval.self_modeling.ground_truth import (
    ResamplingEngine,
)
from prompt_attribution.shared.config import (
    get_domain_for_benchmark,
    load_perturbation,
)

logger = logging.getLogger(__name__)

# ANSI colors for log tags
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"

# GPT-OSS expects this preamble in the system prompt to route reasoning into
# its "analysis" channel. The "Current date" line matches the value baked into
# the model's training data — leave as-is even if the calendar moves on.
GPT_OSS_SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2026-04-13\n\n"
    "Reasoning: high\n\n"
    "# Valid channels: analysis, commentary, final. "
    "Channel must be included for every message."
)


def _short_model_name(model_id: str) -> str:
    """Shorten model ID for filenames and display."""
    name = model_id.split("/")[-1]
    for prefix in ["claude-", "meta-llama-", "Meta-Llama-"]:
        name = name.replace(prefix, "")
    # Truncate to reasonable length
    if len(name) > 40:
        name = name[:40]
    return name.replace(" ", "_")


class SelfModelingRunner:
    """Orchestrates self-modeling evals across models.

    Reuses existing infrastructure:
    - Benchmark loaders from eval/benchmarks/
    - Domain verifiers from eval/domains/
    - safetytooling.InferenceAPI for cloud providers; AsyncOpenAI for vLLM
    """

    def __init__(self, config: SelfModelingConfig):
        self.config = config

        # Resolve domain from benchmark
        self.domain_name = get_domain_for_benchmark(config.benchmark)

        # Load perturbation
        self.perturbation = load_perturbation(self.domain_name, config.perturbation_id)

        # Will be initialized in run()
        self._benchmark: Optional[BaseBenchmark] = None
        self._domain: Optional[BaseDomain] = None
        self._verifier: Optional[BaseVerifier] = None
        self._domain_lang: Optional[DomainLanguage] = None
        self._examples: list[Example] = []
        self._evals: list[BaseSelfModelingEval] = []

        # Phase-2 prompts that were sent per model, keyed by eval_id. Used by
        # the per-group summary writer for debugging / self-modeling.
        self._prompts_sent: dict[str, dict[int, list[str]]] = {}

    def _init_benchmark_and_domain(self) -> None:
        """Initialize benchmark, domain, and verifier."""
        from prompt_attribution.eval.benchmarks import get_benchmark as _get_benchmark
        from prompt_attribution.eval.domains import create_domain

        self._benchmark = _get_benchmark(self.config.benchmark)
        self._domain = create_domain(self.domain_name, self.perturbation)
        self._verifier = self._domain.create_verifier()
        self._domain_lang = build_domain_language(
            self.domain_name, self._benchmark, self.perturbation
        )

    def _load_examples(self) -> list[Example]:
        """Load benchmark examples."""
        return self._benchmark.load_examples(
            n_samples=self.config.n_samples,
            random_seed=self.config.random_seed,
        )

    async def run(self) -> dict:
        """Run all selected evals across all models.

        Returns:
            Dict mapping model -> eval_id -> {metrics, per_example}
        """
        self._init_benchmark_and_domain()
        self._examples = self._load_examples()

        # Get compatible evals
        self._evals = get_compatible_evals(
            self.config.eval_ids,
            self._benchmark,
            self._domain,
            self.perturbation,
        )

        if not self._evals:
            logger.warning(f"{_YELLOW}[WARNING]{_RESET} No compatible evals found")
            return {}

        eval_names = [f"E{e.eval_id}:{e.eval_name}" for e in self._evals]
        logger.info(
            f"{_GREEN}[INFO]{_RESET} Running {len(self._evals)} evals: {', '.join(eval_names)}"
        )
        logger.info(
            f"{_GREEN}[INFO]{_RESET} Benchmark: {self.config.benchmark}, "
            f"Perturbation: {self.config.perturbation_id}, "
            f"Examples: {len(self._examples)}"
        )

        # Setup output directory
        output_path = self.config.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        self._save_config(output_path)

        # Run per model
        all_results = {}
        for model_id in self.config.models:
            logger.info(
                f"\n{_CYAN}[STEP]{_RESET} === Model: {model_id} ==="
            )
            results = await self._run_model(model_id, output_path)
            all_results[model_id] = results


        return all_results

    async def _run_model(
        self,
        model_id: str,
        output_path: Path,
    ) -> dict:
        """Run all evals for a single model.

        Args:
            model_id: Model identifier
            output_path: Base output directory

        Returns:
            Dict mapping eval_id -> {metrics, per_example}
        """
        from dotenv import load_dotenv
        load_dotenv()

        # Determine if using vLLM (local model) or API (cloud providers)
        use_vllm = bool(self.config.vllm_url)

        # Store reasoning traces from thinking models (populated by call_model)
        _last_reasoning: dict[str, str] = {}  # prompt_hash -> reasoning text

        # Auto-detect thinking mode for the model (used by both vLLM and API paths)
        from prompt_attribution.shared.config.model_format import ModelFormat
        _model_fmt = ModelFormat.from_model_name(model_id, max_tokens=self.config.max_tokens or 2048)
        # Override: --no-thinking disables thinking even for auto-detected thinking models
        if self.config.no_thinking and _model_fmt.thinking:
            _model_fmt.enable_thinking = False
            logger.info(f"{_GREEN}[INFO]{_RESET} Thinking explicitly disabled for {model_id}")
        _thinking_extra = _model_fmt.get_thinking_extra_body() or None
        _is_gptoss = "gpt-oss" in model_id.lower()

        if use_vllm:
            # --- vLLM path: OpenAI-compatible client ---
            from openai import AsyncOpenAI

            vllm_client = AsyncOpenAI(
                base_url=self.config.vllm_url,
                api_key="EMPTY",  # vLLM doesn't need a real key
            )
            if _model_fmt.thinking and _model_fmt.enable_thinking:
                logger.info(
                    f"{_GREEN}[INFO]{_RESET} Thinking enabled for {model_id} "
                    f"({'system prompt' if _is_gptoss else f'budget={_model_fmt.thinking_budget}'})"
                )
            logger.info(
                f"{_GREEN}[INFO]{_RESET} Using vLLM at {self.config.vllm_url}"
            )

            async def call_model(
                prompt: str,
                temperature: float = 0.0,
                use_cache: bool = True,
                max_tokens: int = 0,
            ) -> str:
                from prompt_attribution.eval.self_modeling.config import SELF_MODELING_SYSTEM_PROMPT
                # Build system prompt. GPT-OSS routes reasoning through a
                # channel-tagged system prompt rather than `reasoning_effort`,
                # so prepend its expected preamble for that family.
                sys_prompt = "" if self.config.no_system_prompt else SELF_MODELING_SYSTEM_PROMPT
                if _is_gptoss and not self.config.no_thinking:
                    sys_prompt = GPT_OSS_SYSTEM_PROMPT + "\n\n" + SELF_MODELING_SYSTEM_PROMPT
                try:
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    create_kwargs: dict[str, Any] = dict(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                    )
                    if _thinking_extra:
                        create_kwargs["extra_body"] = _thinking_extra
                    mt = max_tokens if max_tokens > 0 else self.config.max_tokens
                    if mt > 0:
                        create_kwargs["max_tokens"] = mt
                    response = await vllm_client.chat.completions.create(**create_kwargs)
                    msg = response.choices[0].message
                    content = msg.content or ""
                    # Save reasoning if present (from --reasoning-parser)
                    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
                    if reasoning:
                        import hashlib
                        key = hashlib.md5(prompt.encode()).hexdigest()[:12]
                        _last_reasoning[key] = reasoning
                    return content
                except Exception as e:
                    logger.error(
                        f"{_RED}[ERROR]{_RESET} vLLM call failed: "
                        f"{type(e).__name__}: {e}"
                    )
                    return ""

            async def call_model_logprobs(
                prompt: str,
                choices: list[str],
                temperature: float = 0.0,
            ) -> dict[str, float]:
                """Get exact logprobs for MCQ choice tokens via completions endpoint.

                Prompt should already include the answer instruction suffix.
                Uses echo=True with one call per choice for exact logprobs.
                """
                logprob_prompt = prompt + "\n\n"

                async def _get_one(letter: str) -> tuple[str, float]:
                    try:
                        resp = await vllm_client.completions.create(
                            model=model_id,
                            prompt=logprob_prompt + letter,
                            max_tokens=0,
                            echo=True,
                            logprobs=1,
                        )
                        tl = resp.choices[0].logprobs.token_logprobs
                        if tl and tl[-1] is not None:
                            return letter, tl[-1]
                        return letter, -100.0
                    except Exception as e:
                        logger.warning(f"Exact logprob failed for '{letter}': {e}")
                        return letter, -100.0

                results = await asyncio.gather(*[_get_one(l) for l in choices])
                return dict(results)

        else:
            # --- API path: safetytooling InferenceAPI ---
            # No logprobs support for cloud API models
            call_model_logprobs = None  # type: ignore[assignment]

            from safetytooling.apis import InferenceAPI
            from safetytooling.data_models import ChatMessage, MessageRole, Prompt
            from prompt_attribution.shared.config.experiment_config import get_provider

            # Detect provider to pass to safetytooling (it can't auto-detect all models)
            try:
                provider = get_provider(model_id)
            except ValueError:
                provider = None

            from prompt_attribution.eval.self_modeling.config import SELF_MODELING_SYSTEM_PROMPT

            if self.config.thinking and "claude" in model_id.lower():
                # Thinking-enabled path: use AsyncAnthropic directly
                # (safetytooling doesn't support thinking params)
                from anthropic import AsyncAnthropic as _AsyncAnthropic
                _think_client = _AsyncAnthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
                _think_sem = asyncio.Semaphore(self.config.max_concurrent)

                async def call_model(
                    prompt: str,
                    temperature: float = 0.0,
                    use_cache: bool = True,
                    max_tokens: int = 0,
                ) -> str:
                    mt = max_tokens if max_tokens > 0 else self.config.max_tokens
                    async with _think_sem:
                        try:
                            budget = mt // 2
                            create_kwargs = dict(
                                model=model_id,
                                max_tokens=mt,
                                system=SELF_MODELING_SYSTEM_PROMPT,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=1,
                                thinking={
                                    "type": "enabled",
                                    "budget_tokens": budget,
                                },
                            )
                            resp = await _think_client.messages.create(**create_kwargs)
                            thinking_text = ""
                            answer_text = ""
                            for block in resp.content:
                                if block.type == "thinking":
                                    thinking_text = block.thinking
                                elif block.type == "text":
                                    answer_text = block.text
                            if thinking_text:
                                return f"<thinking>{thinking_text}</thinking>\n\n{answer_text}"
                            return answer_text or (resp.content[-1].text if resp.content else "")
                        except Exception as e:
                            logger.error(
                                f"{_RED}[ERROR]{_RESET} Thinking API call failed: "
                                f"{type(e).__name__}: {e}"
                            )
                            return ""
            else:
                api = InferenceAPI(anthropic_num_threads=self.config.max_concurrent)
                api_nocache = InferenceAPI(
                    anthropic_num_threads=self.config.max_concurrent, no_cache=True
                )

                # Detect thinking config for API models
                _api_thinking_enabled = (
                    self.config.thinking
                    or (_model_fmt.thinking and _model_fmt.enable_thinking)
                )
                # Pre-compute provider-specific thinking kwargs
                _api_thinking_kwargs: dict[str, Any] = {}
                if _api_thinking_enabled and provider:
                    # If user set a budget override, apply it to the ModelFormat
                    if self.config.thinking_budget_tokens and self.config.thinking_budget_tokens > 0:
                        _model_fmt.thinking_budget = self.config.thinking_budget_tokens
                    elif _model_fmt.thinking_budget == 0:
                        # Default budget for API models that don't have one set
                        _model_fmt.thinking_budget = (self.config.max_tokens or 2048) // 2
                    # Ensure thinking is marked as enabled for get_api_thinking_kwargs
                    _model_fmt.thinking = True
                    _model_fmt.enable_thinking = True
                    _api_thinking_kwargs = _model_fmt.get_api_thinking_kwargs(provider, model_name=model_id)
                if _api_thinking_enabled:
                    logger.info(
                        f"{_GREEN}[INFO]{_RESET} API thinking enabled for {model_id} "
                        f"(provider={provider or 'auto'}, kwargs={list(_api_thinking_kwargs.keys())})"
                    )

                async def call_model(
                    prompt: str,
                    temperature: float = 0.0,
                    use_cache: bool = True,
                    max_tokens: int = 0,
                ) -> str:
                    the_api = api if use_cache else api_nocache
                    mt = max_tokens if max_tokens > 0 else self.config.max_tokens
                    try:
                        call_kwargs = dict(
                            model_id=model_id,
                            prompt=Prompt(
                                messages=[
                                    ChatMessage(role=MessageRole.system, content=SELF_MODELING_SYSTEM_PROMPT),
                                    ChatMessage(role=MessageRole.user, content=prompt),
                                ]
                            ),
                            temperature=temperature,
                            max_tokens=mt,
                        )
                        if provider:
                            call_kwargs["force_provider"] = provider

                        # Add provider-specific thinking params
                        if _api_thinking_enabled and _api_thinking_kwargs:
                            call_kwargs.update(_api_thinking_kwargs)

                        responses = await the_api(**call_kwargs)
                        # Save thinking/reasoning if present
                        thinking = getattr(responses[0], "thinking", "") or ""
                        if thinking:
                            import hashlib
                            key = hashlib.md5(prompt.encode()).hexdigest()[:12]
                            _last_reasoning[key] = thinking
                        return responses[0].completion
                    except Exception as e:
                        logger.error(
                            f"{_RED}[ERROR]{_RESET} API call failed: "
                            f"{type(e).__name__}: {e}"
                        )
                        return ""

        # Shared semaphore across ALL batch calls (prevents flooding vLLM
        # when Phase A runs E5/E6/E7 GT concurrently)
        _shared_sem = asyncio.Semaphore(self.config.max_concurrent)

        # Batch wrapper (shared by both paths)
        async def call_model_batch(
            prompts: list[str],
            temperature: float = 0.0,
            use_cache: bool = True,
            concurrency: int = 0,
            max_tokens: int = 0,
        ) -> list[str]:
            async def _call(p: str) -> str:
                async with _shared_sem:
                    return await call_model(p, temperature, use_cache, max_tokens)

            return await asyncio.gather(*[_call(p) for p in prompts])

        # GT and Phase 2 use the same model (self-self-modeling only)
        call_model_gt = call_model  # type: ignore[assignment]
        call_model_gt_batch = call_model_batch  # type: ignore[assignment]
        call_model_phase2_batch = call_model_batch  # type: ignore[assignment]

        # Setup resampling engine
        model_short = _short_model_name(model_id)
        cache_dir = output_path / ".cache" / f"{self.config.benchmark}_{model_short}"
        logprobs_fn = call_model_logprobs if use_vllm else None

        engine = ResamplingEngine(
            call_model_fn=call_model_gt,
            call_model_batch_fn=call_model_gt_batch,
            benchmark=self._benchmark,
            verifier=self._verifier,
            domain=self._domain,
            n_resample=self.config.n_resample,
            resample_temperature=self.config.resample_temperature,
            cache_dir=cache_dir,
            call_model_logprobs_fn=logprobs_fn,
        )

        # Determine what resampling is needed
        needs_baseline = any(e.capabilities.needs_baseline_resamples for e in self._evals)
        needs_lever = any(e.capabilities.needs_lever_resamples for e in self._evals)
        needs_flip_gt = any(e.capabilities.needs_flip_gt for e in self._evals)
        needs_logprobs = any(e.capabilities.needs_logprobs for e in self._evals) and use_vllm

        # Step 1: Resample (shared across evals)
        logger.info(f"{_CYAN}[STEP]{_RESET} Resampling ground truth...")
        t0 = time.time()
        resampled = await engine.resample_all(
            self._examples,
            self.perturbation,
            needs_baseline=needs_baseline,
            needs_lever=needs_lever,
            needs_flip_gt=needs_flip_gt,
            needs_logprobs=needs_logprobs,
        )
        logger.info(
            f"{_GREEN}[INFO]{_RESET} Resampling done in {time.time() - t0:.1f}s"
        )

        # Step 2: Run evals in 3 parallel phases
        #   Phase A: Compute GT for ALL evals concurrently
        #   Phase B: Build ALL prompts, run in ONE batch
        #   Phase C: Score ALL evals concurrently
        import inspect

        model_results = {}
        self._prompts_sent[model_id] = {}

        # --- Prepare per-eval GT kwargs ---
        eval_gt_kwargs: dict[int, dict[str, Any]] = {}
        active_evals: list[BaseSelfModelingEval] = []

        for eval_obj in self._evals:
            gt_kwargs: dict[str, Any] = {
                "perturbation": self.perturbation,
                "domain_lang": self._domain_lang,
                "call_model_batch_fn": call_model_gt_batch,
                "call_model_fn": call_model_gt,
                "benchmark": self._benchmark,
                "n_resample": self.config.n_resample,
                "resample_temperature": self.config.resample_temperature,
                "decomposer_cache_dir": output_path / ".cache" / "decomposer",
            }
            if eval_obj.capabilities.needs_multiple_perturbations:
                if self.config.eval6_perturbation_ids:
                    gt_kwargs["perturbation_configs"] = [
                        load_perturbation(self.domain_name, pid)
                        for pid in self.config.eval6_perturbation_ids
                    ]
                else:
                    logger.warning(
                        f"{_YELLOW}[WARNING]{_RESET} Eval {eval_obj.eval_id} skipped: "
                        "eval6_perturbation_ids not set"
                    )
                    continue
            eval_gt_kwargs[eval_obj.eval_id] = gt_kwargs
            active_evals.append(eval_obj)

        # --- Phase A: Compute GT for ALL evals concurrently ---
        logger.info(
            f"{_CYAN}[STEP]{_RESET} Phase A: Computing GT for "
            f"{len(active_evals)} evals concurrently..."
        )
        t_gt = time.time()

        async def _compute_gt(eval_obj: BaseSelfModelingEval) -> Any:
            gt_kw = eval_gt_kwargs[eval_obj.eval_id]
            gt_result = eval_obj.compute_ground_truth(
                self._examples, resampled, self._verifier, self._domain,
                **gt_kw,
            )
            if inspect.isawaitable(gt_result):
                return await gt_result
            return gt_result

        gt_tasks = [_compute_gt(e) for e in active_evals]
        gt_outcomes = await asyncio.gather(*gt_tasks, return_exceptions=True)

        gt_map: dict[int, Any] = {}
        for eval_obj, outcome in zip(active_evals, gt_outcomes):
            eid = eval_obj.eval_id
            if isinstance(outcome, Exception):
                logger.error(
                    f"  {_RED}[ERROR]{_RESET} GT for E{eid} failed: "
                    f"{type(outcome).__name__}: {outcome}",
                    exc_info=outcome,
                )
                model_results[eid] = {"metrics": {"error": str(outcome)}, "per_example": []}
            else:
                gt_map[eid] = outcome

        logger.info(
            f"  {_GREEN}[INFO]{_RESET} GT done in {time.time() - t_gt:.1f}s "
            f"({len(gt_map)} succeeded)"
        )

        # --- Phase B: Build ALL prompts, run in ONE batch ---
        all_prompts: list[str] = []
        prompt_slices: dict[int, tuple[int, int]] = {}  # eid -> (start, end)

        for eval_obj in active_evals:
            eid = eval_obj.eval_id
            if eid not in gt_map:
                continue
            gt = gt_map[eid]

            prompt_kwargs: dict[str, Any] = {"resampled_data": resampled}
            if eval_obj.capabilities.needs_multiple_perturbations:
                prompt_kwargs["perturbation_configs"] = eval_gt_kwargs[eid].get(
                    "perturbation_configs", []
                )

            prompts = eval_obj.build_phase2_prompts(
                self._examples, self._benchmark, self._domain_lang,
                gt, self.perturbation, **prompt_kwargs,
            )
            self._prompts_sent[model_id][eid] = prompts
            start = len(all_prompts)
            all_prompts.extend(prompts)
            prompt_slices[eid] = (start, start + len(prompts))

        if all_prompts:
            logger.info(
                f"{_CYAN}[STEP]{_RESET} Phase B: Running inference "
                f"({len(all_prompts)} prompts across {len(prompt_slices)} evals)..."
            )
            t_inf = time.time()
            all_responses = await call_model_phase2_batch(
                all_prompts,
                temperature=self.config.phase2_temperature,
                use_cache=True,
            )
            logger.info(
                f"  {_GREEN}[INFO]{_RESET} Inference done in {time.time() - t_inf:.1f}s"
            )
        else:
            all_responses = []

        # --- Phase C: Score ALL evals concurrently ---
        logger.info(
            f"{_CYAN}[STEP]{_RESET} Phase C: Scoring {len(prompt_slices)} evals..."
        )
        t_score = time.time()

        score_kwargs = {
            "verifier": self._verifier,
            "domain": self._domain,
            "call_model_batch_fn": call_model_gt_batch,
            "benchmark": self._benchmark,
            "perturbation": self.perturbation,
            "resampled_data": resampled,
        }

        async def _score_eval(
            eval_obj: BaseSelfModelingEval,
        ) -> tuple[int, list[dict], dict]:
            eid = eval_obj.eval_id
            start, end = prompt_slices[eid]
            responses = all_responses[start:end]
            gt = gt_map[eid]

            score_result = eval_obj.score(
                self._examples, responses, gt, **score_kwargs,
            )
            if inspect.isawaitable(score_result):
                scored = await score_result
            else:
                scored = score_result
            metrics = eval_obj.aggregate_metrics(scored)
            return eid, scored, metrics

        score_evals = [e for e in active_evals if e.eval_id in prompt_slices]
        score_outcomes = await asyncio.gather(
            *[_score_eval(e) for e in score_evals],
            return_exceptions=True,
        )

        for eval_obj, outcome in zip(score_evals, score_outcomes):
            eid = eval_obj.eval_id
            if isinstance(outcome, Exception):
                logger.error(
                    f"  {_RED}[ERROR]{_RESET} Scoring E{eid} failed: "
                    f"{type(outcome).__name__}: {outcome}",
                    exc_info=outcome,
                )
                model_results[eid] = {"metrics": {"error": str(outcome)}, "per_example": []}
            else:
                _, scored, metrics = outcome
                model_results[eid] = {"metrics": metrics, "per_example": scored}
                logger.info(
                    f"  {_MAGENTA}[EVAL]{_RESET} E{eid}: {json.dumps(metrics, default=str)}"
                )

        logger.info(
            f"  {_GREEN}[INFO]{_RESET} Scoring done in {time.time() - t_score:.1f}s"
        )

        # Step 3: Write outputs
        model_output_dir = output_path / model_short
        model_output_dir.mkdir(parents=True, exist_ok=True)
        self._write_model_results(model_id, model_results, model_output_dir)

        # Save prompts_sent to disk so HTML can be regenerated without re-running
        prompts_path = model_output_dir / "prompts_sent.json"
        with open(prompts_path, "w") as f:
            json.dump(
                {str(eid): prompts for eid, prompts in self._prompts_sent.get(model_id, {}).items()},
                f,
            )

        # Save reasoning traces from thinking models (if any)
        if _last_reasoning:
            reasoning_path = model_output_dir / "reasoning_traces.json"
            with open(reasoning_path, "w") as f:
                json.dump(_last_reasoning, f, indent=2)
            logger.info(
                f"{_GREEN}[INFO]{_RESET} Saved {len(_last_reasoning)} reasoning traces: {reasoning_path}"
            )

        # Step 4: Generate HTML debug viewer

        return model_results

    def _save_config(self, output_path: Path) -> None:
        """Save experiment config to YAML and CLI command to command.txt."""
        config_path = output_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

        # Save CLI command for reproducibility
        command_path = output_path / "command.txt"
        with open(command_path, "w") as f:
            f.write(" ".join(sys.argv) + "\n")

    def _write_model_results(
        self,
        model_id: str,
        results: dict[int, dict],
        output_dir: Path,
    ) -> None:
        """Write per-eval CSV files and summary JSON."""
        # Summary JSON
        summary = {
            "model": model_id,
            "benchmark": self.config.benchmark,
            "perturbation": self.config.perturbation_id,
            "evals": {},
        }

        for eid, result in results.items():
            metrics = result["metrics"]
            per_example = result["per_example"]

            strict = self._compute_strict_metrics(eid, metrics, per_example)
            summary["evals"][str(eid)] = {**metrics, "strict": strict}

            # Write CSV
            eval_obj = next((e for e in self._evals if e.eval_id == eid), None)
            eval_name = eval_obj.eval_name if eval_obj else f"eval{eid:02d}"
            csv_path = output_dir / f"eval{eid:02d}_{eval_name.lower().replace(' ', '_')}.csv"

            if per_example:
                # Filter out raw_response fields for CSV (keep them for HTML)
                csv_fields = [
                    k for k in per_example[0].keys()
                    if not k.startswith("raw_")
                ]
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(per_example)

        # Write full results JSON (preserves all data types for HTML regeneration)
        full_results_path = output_dir / "full_results.json"
        with open(full_results_path, "w") as f:
            json.dump(
                {str(eid): result for eid, result in results.items()},
                f, indent=2, default=str,
            )

        # Write summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"{_GREEN}[INFO]{_RESET} Results saved to {output_dir}"
        )

    @staticmethod
    def _compute_strict_metrics(
        eid: int, metrics: dict, per_example: list[dict]
    ) -> dict:
        """Compute 'strict' metrics where parse failures count as 0.

        The default metrics skip unparseable responses (n_valid < n_total).
        Strict mode counts every example: parse failure = wrong answer (accuracy)
        or worst-case error (MSE=1).

        Returns a dict with the same keys as metrics but with strict values.
        """
        n_total = metrics.get("n_total", len(per_example))
        if n_total == 0:
            return {}

        strict: dict[str, Any] = {"n_total": n_total}

        if eid in (1, 6, 7):
            # Accuracy evals: n_correct / n_total
            n_valid = metrics.get("n_valid", 0)
            acc = metrics.get("accuracy")
            n_correct = int(round(acc * n_valid)) if acc is not None and n_valid else 0
            strict["accuracy"] = n_correct / n_total
            strict["n_parsed"] = n_valid

        elif eid == 2:
            # Similarity: sum valid sims / n_total
            n_valid = metrics.get("n_valid", 0)
            mean_sim = metrics.get("mean_similarity")
            total_sim = mean_sim * n_valid if mean_sim is not None and n_valid else 0.0
            strict["mean_similarity"] = total_sim / n_total
            strict["n_parsed"] = n_valid

        elif eid in (3, 4, 5, 9):
            # MSE evals: failed = mse=1
            n_valid = metrics.get("n_valid", 0)
            mse = metrics.get("mse")
            sum_mse = mse * n_valid if mse is not None and n_valid else 0.0
            n_failed = n_total - n_valid
            strict["mse"] = (sum_mse + n_failed * 1.0) / n_total
            strict["n_parsed"] = n_valid

        elif eid == 8:
            # Flip accuracy: failed = no flip
            n_valid = metrics.get("n_valid", 0)
            flip_acc = metrics.get("flip_accuracy")
            n_flipped = int(round(flip_acc * n_valid)) if flip_acc is not None and n_valid else 0
            strict["flip_accuracy"] = n_flipped / n_total
            strict["n_parsed"] = n_valid

        elif eid == 10:
            # Margin: failed = mse=1; Second: failed = wrong
            n_margin = metrics.get("n_margin", 0)
            n_second = metrics.get("n_second", 0)
            mse_margin = metrics.get("mse_margin")
            acc_second = metrics.get("accuracy_second")
            # Can't easily recover n_correct for second from aggregate,
            # so just report the values assuming all non-parsed are failures
            strict["mse_margin"] = mse_margin  # already over n_margin
            strict["accuracy_second"] = acc_second  # already over n_second
            strict["n_margin"] = n_margin
            strict["n_second"] = n_second

        return strict

