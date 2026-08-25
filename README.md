# Self-Modeling Eval

This is the official repository implementing the self-modeling benchmark
(E1–E10) from Evaluating and Improving LLM Self-Modeling at the 2026
Conference on Empirical Methods in Natural Language Processing (EMNLP 2026).

A 10-eval self-modeling benchmark (E1–E10) that asks a language model to
predict aspects of its own behavior on perturbed prompts — will my answer
flip? what would my new answer be? which of these three perturbations
matters most? — and scores the prediction against a resampled ground truth.
The test set is sampled across four benchmarks (GSM8K, HumanEval, BBQ,
WildGuardTest) and four perturbation families (math / coding / safety /
fairness).

## Quick start

```bash
uv sync                                                       # 1. install
uv run python scripts/setup/apply_safetytooling_patches.py    # 2. patch safetytooling
cp .env.example .env && $EDITOR .env                          # 3. add your API keys

# 4. Smoke run on a small sample to verify the install:
uv run python scripts/experiments/run_self_modeling_mixed.py \
    --models claude-haiku-4-5-20251001 \
    --n-samples 5 --n-resample 1 --seed 42 \
    --thinking --budget-tokens 4096
```

## Setup

### 1. Install

```bash
uv sync
```

Installs `safetytooling`, `anthropic`, `openai`, `datasets`, `math-verify`,
and a handful of transitive deps (~1.5 GB venv).

### 2. Apply the safetytooling overlay

The pinned `safetytooling` build needs a few small edits before all
providers work end-to-end (extracting `thinking` / `reasoning` blocks for
Anthropic / OpenAI / Together / Gemini, plus a fast async-SDK path for
`gemini-2.5+` / `gemini-3+`). The patched files are vendored under
`patches/safetytooling/`; overlay them after every install:

```bash
uv run python scripts/setup/apply_safetytooling_patches.py
```

Re-run this after any `uv sync` — the patches live inside the installed
package directory and are lost when the package is reinstalled.

### 3. API keys

```bash
cp .env.example .env
```

Only the keys for the providers you plan to call are required:

| Provider | Env var | Model-id prefix |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-*` |
| OpenAI | `OPENAI_API_KEY` | `gpt-*`, `o1-*`, `o3*`, `o4*` |
| Google Gemini | `GOOGLE_API_KEY` | `gemini-*` |
| Together AI | `TOGETHER_API_KEY` | `meta-llama/*`, `Qwen/*`, `deepseek-ai/*`, `mistralai/*` |
| OpenRouter | `OPENROUTER_API_KEY` | `moonshotai/*`, `x-ai/*` |
| Hugging Face | `HF_TOKEN` | Dataset downloads (GSM8K / BBQ / HumanEval / WildGuardTest) |

The provider is auto-detected from the model id; vLLM is selected by
passing `--vllm-url` regardless of the id.

### API models

```bash
for seed in 42 43 44 45 46; do
    uv run python scripts/experiments/run_self_modeling_mixed.py \
        --models <MODEL_ID> \
        --n-samples 100 --n-resample 1 --seed $seed \
        --thinking --budget-tokens 4096 \
        --max-concurrent 20
done
```

`<MODEL_ID>` examples — the `--thinking` flag is interpreted per provider:

| Model id | Provider | What `--thinking` does |
|---|---|---|
| `claude-opus-4-5-20251101`     | Anthropic  | Extended thinking, `budget_tokens=4096` |
| `claude-sonnet-4-5-20250929`   | Anthropic  | Extended thinking, `budget_tokens=4096` |
| `claude-haiku-4-5-20251001`    | Anthropic  | Extended thinking, `budget_tokens=4096` |
| `gpt-5.4`, `gpt-5.4-mini`      | OpenAI     | `reasoning_effort=high` |
| `o4-mini`                      | OpenAI     | `reasoning_effort=high` |
| `gpt-4o`                       | OpenAI     | (non-reasoning model — flag is ignored) |
| `gemini-3-flash-preview`       | Gemini     | `thinking_budget=4096` |
| `gemini-3.1-pro-preview`       | Gemini     | `thinking_budget=4096` |
| `moonshotai/Kimi-K2.5`         | OpenRouter | Provider-side thinking |
| `x-ai/grok-4.20`               | OpenRouter | Provider-side thinking |

Drop `--thinking --budget-tokens 4096` for non-thinking models.

### vLLM models

Launch the server (vLLM 0.19+ recommended — it enforces
`thinking_token_budget`, which the eval relies on for Qwen3.5):

```bash
# Non-thinking model (Llama, Mistral, …)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000 \
    --dtype auto --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 --enable-prefix-caching

# Thinking model — add a reasoning parser:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --port 8000 \
    --dtype auto --gpu-memory-utilization 0.85 \
    --max-num-seqs 64 --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
    --trust-remote-code
```

Reasoning parser per family:

| Model family | `--reasoning-parser` |
|---|---|
| `Qwen/Qwen3*`, `Qwen/Qwen3.5*` | `qwen3` |
| `deepseek-ai/DeepSeek-R1*`, `DeepSeek-V3.1` | `deepseek_r1` |
| `openai/gpt-oss-*` | `openai_gptoss` |
| `meta-llama/Llama-*` and other non-thinking | (none) |

Then run the eval against the server:

```bash
for seed in 42 43 44 45 46; do
    uv run python scripts/experiments/run_self_modeling_mixed.py \
        --models meta-llama/Llama-3.1-8B-Instruct \
        --n-samples 100 --n-resample 1 --seed $seed \
        --vllm-url http://HOST:PORT/v1 --max-concurrent 50
done
```

Thinking mode is auto-detected from the model id (`qwen3`, `deepseek`,
`gpt-oss`, `kimi` → on). Override with `--no-thinking` to ablate, or
`--thinking --budget-tokens N` to set a specific budget.

## CLI reference

| flag | purpose |
|---|---|
| `--models ID [ID ...]`     | model id(s) to evaluate (space-separated for multiple) |
| `--n-samples N`            | examples per eval (default 100) |
| `--n-resample N`           | resampling count for GT (default 5; published runs use 1) |
| `--seed N`                 | random seed for test-set sampling and resampling |
| `--vllm-url URL`           | base URL of an OpenAI-compatible vLLM server |
| `--max-concurrent N`       | concurrent calls (vLLM ~50, cloud APIs ~20) |
| `--thinking`               | enable extended thinking / reasoning on the provider |
| `--budget-tokens N`        | thinking budget where applicable (Anthropic min 1024) |
| `--no-thinking`            | force-disable thinking on a model that auto-enables it |
| `--no-system-prompt`       | ablation: skip the self-modeling system prompt |
| `--eval-ids 1 2 3 ...`     | run a subset of E1–E10 (default all 10) |
| `--phase2-temp F`          | sampling temperature for the self-modeling query |
| `--resample-temp F`        | sampling temperature for the GT-resampling pass |
| `--output-dir PATH`        | base output directory (default `outputs/results/mixed_self_modeling`) |

## Output layout

```
outputs/results/mixed_self_modeling/
  mixed_<TIMESTAMP>/                           # one timestamped dir per run
    test_set_manifest.json                     # examples + perturbations sampled
    command.txt                                # CLI invocation
    scores_<MODEL_SHORT>.json                  # ← the main result
    <benchmark>_<perturbation>/                # per-group breakdown
      <benchmark>_<perturbation>_<TS>/
        <MODEL_SHORT>/
          summary.json                         # per-eval metrics for the group
          eval01_flip_prediction.csv           # per-example predictions vs GT
          eval02_output_prediction.csv
          ...
```

Useful keys in `scores_<MODEL_SHORT>.json`:

- `strict_overall` — strict E1–E10 macro-average (NA counts as wrong)
- `lenient_overall` — same, skipping NA responses
- `strict_per_eval` / `lenient_per_eval` — per-eval breakdown plus baselines
- `strict_skill_overall` / `lenient_skill_overall` — score − baseline

## Layout

```
src/prompt_attribution/
  eval/
    self_modeling/   # E1–E10, runner, GT engine, parsers
    benchmarks/      # GSM8K, HumanEval, BBQ, WildGuardTest loaders
    domains/         # math / coding / safety / fairness verifiers
  shared/
    config/          # PerturbationConfig, registry, model_format
patches/safetytooling/                          # files overlaid on installed safetytooling
scripts/
  experiments/run_self_modeling_mixed.py       # entry point
  setup/apply_safetytooling_patches.py         # post-install patcher
```

## Citation

```bibtex
@inproceedings{zeng2026evaluating,
  title={Evaluating and Improving {LLM} Self-Modeling},
  author={Siqi Zeng and Andre N. Assis and Rowan Wang},
  booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026},
  url={https://openreview.net/forum?id=GdUcIPKke1}
}
```