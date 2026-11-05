# SCDBench

This repository contains the evaluation code and model outputs for **SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers**, accepted at the **NeurIPS 2026 Evaluations and Datasets Track**.

**Authors:** Kaihua Qin, Dawn Song, and Arthur Gervais.

**[Paper](https://arxiv.org/abs/2605.29059)** | **[Dataset](https://doi.org/10.7910/DVN/IPPTIX)**

SCDBench evaluates decompilation from EVM bytecode to Solidity through four cumulative stages: format completeness, compilability, ABI recovery, and semantic consistency. The original benchmark contains 600 real-world contracts, with 200 contracts in each of three complexity strata, and 227,383 semantic test cases covering 14,553 callable entries. A second, non-overlapping 600-contract sample supports the sampling-stability analysis.

## Dataset

The benchmark dataset is publicly available on [Harvard Dataverse](https://doi.org/10.7910/DVN/IPPTIX) under CC BY 4.0.

Download `scdbench.jsonl` and either place it at `../dataset/scdbench.jsonl` relative to the repository root, or pass its location through `--dataset` and `--testcases`. The examples below use the default sibling `dataset` directory.

The second sample is `scdbench_second_sample.jsonl`, containing 226,138 checkpoints over 14,578 callable entries. It supplements rather than replaces the original dataset. Use its path for both `--dataset` and `--testcases`, and keep its outputs in `runs/second_sample/`.

## Getting Started

```bash
git clone https://github.com/KaihuaQin/SCDBench.git
cd SCDBench
```

Run all commands below from the repository root.

## Layout

- `../dataset/scdbench.jsonl`: default expected location for the benchmark dataset.
- `scripts/run_openrouter_benchmark.py`: zero-shot decompilation through OpenRouter.
- `scripts/run_openrouter_compilation_repair.py`: zero-shot compilation repair through OpenRouter.
- `scripts/evaluate_model_outputs.py`: format, compilation, and ABI evaluation for one prediction file.
- `scripts/run_first_three_evaluations.py`: batch runner for format, compilation, and ABI evaluation.
- `scripts/run_foundry_semantic.py`: Foundry-based semantic replay.
- `scripts/run_semantic_model_suite.py`: batch runner for semantic replay before and after repair.
- `scripts/foundry_compile.py`: Solidity compilation helper used by the evaluation scripts.
- `scripts/foundry_semantic/`: Foundry harness used for semantic replay.
- `runs/openrouter_*.jsonl`: sanitized zero-shot decompilation outputs.
- `runs/openrouter_repair_*.jsonl`: sanitized zero-shot repair outputs. These files contain repair attempts. The semantic batch runner merges them with the original outputs when building after-repair predictions.
- `runs/second_sample/`: sanitized initial and repair outputs for the second sample.
- `experiments/`: analysis code, compact recorded inputs, and reproduction instructions.

## Experiments

The following packages reproduce the paper's analyses from stored experiment data.
Their READMEs distinguish offline table reproduction from rerunning experiments
that require compilation, an archive node, or model API calls.

| Analysis | Code and instructions |
|---|---|
| Second-sample initial and repaired results | [Second sample](experiments/second_sample/README.md) |
| Complexity correlations and adjusted associations | [Complexity validation](experiments/difficulty_validation/README.md) |
| Function-selector and event-signature ablation | [Signature ablation](experiments/signature_ablation/README.md) |
| Checkpoint effectiveness and coverage sensitivity | [Semantic effectiveness](experiments/semantic_effectiveness/README.md) |
| Deployed-versus-verified population statistics | [External validity](experiments/external_validity/README.md) |
| ItyFuzz coverage comparison | [Fuzzing comparison](experiments/ityfuzz_comparison/README.md) |
| Sequential replay timing and budgeted prefixes | [Replay timing](experiments/semantic_checkpoint_timing/README.md) |

For example, reproduce and verify all five second-sample tables without API calls:

```bash
python experiments/second_sample/analyze.py --check
```

Statistical analyses additionally use the dependencies in `requirements-analysis.txt`:

```bash
python -m pip install -r requirements-analysis.txt
```

Reproduce all analyses and run their regression tests with:

```bash
python experiments/verify.py
```

This uses only packaged inputs and makes no network calls. The complexity
analysis includes 2,000 bootstrap replicates. Use `--tests-only` for a shorter
test run without regenerating the analysis reports.

## Requirements

Use Python 3.10 or newer. Install Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Semantic replay also requires Foundry tools on `PATH`:

```bash
forge --version
cast --version
```

The OpenRouter scripts load `.env` from the current working directory. For API runs, create `.env` in the repository root with:

```bash
OPENROUTER_API_KEY=...
```

Semantic replay uses a fixed fork block, `24136053`, and requires an archive RPC endpoint:

```bash
ARCHIVE_RPC_URL=...
```

## Zero-Shot Decompilation

Example for GPT-5.3-Codex with high reasoning:

```bash
python scripts/run_openrouter_benchmark.py \
  --dataset ../dataset/scdbench.jsonl \
  --model openai/gpt-5.3-codex \
  --output runs/openrouter_gpt53_codex_reasoning_high.jsonl \
  --extra-request-json '{"reasoning":{"enabled":true,"effort":"high"}}' \
  --max-completion-tokens 65536 \
  --temperature 0 \
  --concurrency 10 \
  --yes
```

Example for GLM-5 without reasoning:

```bash
python scripts/run_openrouter_benchmark.py \
  --dataset ../dataset/scdbench.jsonl \
  --model z-ai/glm-5 \
  --output runs/openrouter_glm5_reasoning_none.jsonl \
  --extra-request-json '{"reasoning":{"effort":"none"}}' \
  --max-completion-tokens 8192 \
  --temperature 0 \
  --concurrency 10 \
  --yes
```

The script writes JSONL incrementally and resumes by `benchmark_id`, so interrupted runs can be restarted with the same output path.

## Format, Compilation, and ABI Evaluation

To evaluate all included decompilation and repair outputs:

```bash
python scripts/run_first_three_evaluations.py \
  --dataset ../dataset/scdbench.jsonl \
  --runs-dir runs \
  --concurrency 10
```

This produces files such as:

- `runs/eval_first3_gpt53_codex_reasoning_high.jsonl`
- `runs/eval_first3_gpt53_codex_reasoning_high.summary.json`
- `runs/eval_first3_repair_gpt53_codex_reasoning_high.jsonl`
- `runs/eval_first3_repair_gpt53_codex_reasoning_high.summary.json`

To evaluate one prediction file directly:

```bash
python scripts/evaluate_model_outputs.py \
  --dataset ../dataset/scdbench.jsonl \
  --predictions runs/openrouter_gpt53_codex_reasoning_high.jsonl \
  --output runs/eval_first3_gpt53_codex_reasoning_high.jsonl \
  --summary runs/eval_first3_gpt53_codex_reasoning_high.summary.json \
  --concurrency 10
```

## Zero-Shot Compilation Repair

Repair uses the compilation errors produced by `evaluate_model_outputs.py`.

Example for GLM-5 without reasoning:

```bash
python scripts/run_openrouter_compilation_repair.py \
  --evaluations runs/eval_first3_glm5_reasoning_none.jsonl \
  --predictions runs/openrouter_glm5_reasoning_none.jsonl \
  --output runs/openrouter_repair_glm5_reasoning_none.jsonl \
  --model z-ai/glm-5 \
  --extra-request-json '{"reasoning":{"effort":"none"}}' \
  --max-completion-tokens 65536 \
  --temperature 0 \
  --concurrency 5 \
  --yes
```

The repair script writes only repaired attempts. Downstream batch evaluation treats a contract as after-repair by using the repaired output when present and the original output otherwise.

## Semantic Replay

Semantic replay first prepares ABI-matched replay targets, then runs the stored checkpoint cases with Foundry. It evaluates functions whose signatures are recovered by the model.

To run all included model outputs before and after repair:

```bash
python scripts/run_semantic_model_suite.py \
  --dataset ../dataset/scdbench.jsonl \
  --testcases ../dataset/scdbench.jsonl \
  --runs-dir runs \
  --prepare-concurrency 10 \
  --prepare-jobs 4 \
  --replay-jobs 4
```

Outputs are written under:

- `runs/semantic_prep/`: compilation and ABI preparation for semantic replay.
- `runs/semantic_targets/`: ABI-matched replay targets.
- `runs/semantic/`: semantic replay results.
- `runs/semantic_logs/`: Foundry logs.

To replay one prepared target file directly:

```bash
python scripts/run_foundry_semantic.py replay \
  --targets runs/semantic_targets/semantic_targets_gpt53_codex_reasoning_high_before.json \
  --testcases ../dataset/scdbench.jsonl \
  --output runs/semantic/semantic_replay_gpt53_codex_reasoning_high_before.json \
  --fork-block 24136053 \
  --matched-functions-only
```

## Notes

- The stored model outputs are included for reproducibility and to avoid re-running paid model calls. Provider response IDs, timestamps, encrypted reasoning blobs, and internal prompt hashes are removed from the release files.
- Do not commit `.env` files or API keys.
- `py-solc-x` installs Solidity compiler versions on demand. The scripts use a local temporary directory under the repository root for solc downloads.

## License

The original SCDBench evaluation and analysis code is released under the [MIT License](LICENSE). The benchmark dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The code license does not relicense third-party contract sources or stored model outputs, which remain subject to any applicable terms.

## Citation

Please cite the paper when using SCDBench:

```bibtex
@inproceedings{qin2026scdbench,
  title = {{SCDBench}: A Benchmark for {LLM}-Based Smart Contract Decompilers},
  author = {Qin, Kaihua and Song, Dawn and Gervais, Arthur},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://arxiv.org/abs/2605.29059}
}
```

The dataset can also be cited using its DOI: [10.7910/DVN/IPPTIX](https://doi.org/10.7910/DVN/IPPTIX).
