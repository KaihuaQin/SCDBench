# Second-Sample Replication

This experiment reproduces the five result tables for the second 600-contract
sample. The model settings are Opus 4.7 high, GPT-5.3-Codex high, GLM-5 high,
and GLM-5 instruct, each with initial output and one-step compilation repair.

## Reproduce the Tables Offline

From the repository root:

```bash
python experiments/second_sample/analyze.py --check
```

This needs only Python's standard library. It does not call a model, compiler,
or archive node. It recomputes the metrics from individual contract evaluations
and function-level replay outcomes, writes `results/metrics.json` and
`results/tables.md`, and checks every reported value against the archived
experiment summaries. The reference summaries are used only for validation.

The metric functions are retained from the original experiment. All 600
contracts remain in the denominator. ABI recall uses all ground-truth ABI
functions and semantic consistency uses all 14,578 callable checkpoint groups.
Initial format eligibility is preserved when scoring repaired outputs.

## Stored Inputs

- `../../runs/second_sample/`: four initial prediction files and four repair files.
- `data/contracts.json`: contract IDs, complexity groups, ABI counts, and checkpoint signatures.
- `data/<model>/*.evaluation.jsonl`: format, compilation, and ABI results.
- `data/<model>/*.replay.json`: function-level semantic replay results.
- `data/reference_metrics.json`: archived results for independent output checks.
- `data/manifest.json`: source hashes and packaged-file hashes.

Prediction files retain the final response for each contract, including model
content, finish reason, request settings, and token/cost usage. Transport retry
duplicates, provider IDs, private paths, and encrypted reasoning are not part
of the release. Compiler diagnostics replace local filesystem paths with
`<local-path>`. Model-generated content is unchanged.

The replay summaries omit repetitive execution logs and per-call payloads.
They preserve every function result needed for the published metrics. Complete
ground-truth checkpoints are in the dataset, and the scripts below can rerun
the comparisons from the released predictions.

## Reevaluate the Predictions

Download `scdbench_second_sample.jsonl` from
https://doi.org/10.7910/DVN/IPPTIX and place it in `../dataset/` relative to
the repository root. Install the dependencies and Foundry described in the
main README. Semantic replay requires a suitable Ethereum archive endpoint.

```bash
python scripts/run_first_three_evaluations.py \
  --dataset ../dataset/scdbench_second_sample.jsonl \
  --runs-dir runs/second_sample

python scripts/run_semantic_model_suite.py \
  --dataset ../dataset/scdbench_second_sample.jsonl \
  --testcases ../dataset/scdbench_second_sample.jsonl \
  --runs-dir runs/second_sample

python scripts/summarize_perfect_semantic_contracts.py \
  --dataset ../dataset/scdbench_second_sample.jsonl \
  --runs-dir runs/second_sample
```

These commands use stored model outputs, so they do not incur model API fees.
They do require local compilation and, for replay, archive-node access. New
evaluation files are separate from the frozen `data/` inputs used by the
offline table check.

## Inference Settings

| Model setting | Initial maximum completion tokens | Repair maximum completion tokens |
|---|---:|---:|
| Opus 4.7 high | 65,536 | 65,536 |
| GPT-5.3-Codex high | 65,536 | 65,536 |
| GLM-5 high | 32,768 | 65,536 |
| GLM-5 instruct | 8,192 | 65,536 |

Temperature is zero. Provider routing was not pinned. Exact request parameters
are retained in each prediction record. New API calls may produce different
outputs because model serving can change. Use the stored outputs to reproduce
the reported results.

## Recreate This Release from the Experiment Workspace

`export_inputs.py` is the offline packaging script. It requires `jq` and access
to the original experiment workspace, not needed for normal reproduction:

```bash
python experiments/second_sample/export_inputs.py --source-root /path/to/experiment-workspace
```

It selects the public fields, validates transport completion, records hashes,
and extracts the minimal stage and replay inputs without rerunning inference.
