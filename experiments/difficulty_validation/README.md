# Complexity Validation

This analysis reproduces the complexity-score associations reported in the
paper. It uses the original 600-contract sample and the four model
settings. Format completeness is measured before repair. Compilation, ABI F1,
and semantic consistency use the repaired outcomes and cumulative denominators.

## Run Offline

From the repository root:

```bash
python -m pip install -r experiments/difficulty_validation/requirements.txt
python experiments/difficulty_validation/analyze.py
python experiments/difficulty_validation/test_analyze.py
python experiments/difficulty_validation/check_results.py
```

The pinned numerical dependencies were tested with Python 3.13.5.

The default run uses 2,000 contract-bootstrap replicates with seed `20260724`
and writes `results.json` and `report.md` beside the script. It does not call a
model, compile a contract, or contact an Ethereum node. The regression controls
for model, prompt length, and public-function count. The original analysis
functions for Spearman correlations, component associations, OLS, clustered
standard errors, and bootstrap intervals are retained.

## Inputs

`inputs/observations.json` contains 600 contract metadata records and 2,400
contract-model observations. It includes the construction-time complexity score
and its five components, callable-entry counts, prompt lengths, and the observed
format, compilation, ABI, and semantic outcomes. Scores were not fitted to these
outcomes. Missing provider token counts are represented by JSON `null`.

The file is a projection of the stored evaluations, not model predictions or
synthetic data. Its provenance lists the SHA-256 hashes of the source files.
`expected_results.json` preserves the reference analysis for regression checks.
Output paths and file-size metadata differ from that internal reference.

The construction scores are additional metadata because the public benchmark
JSONL intentionally contains only the fields needed for inference and replay.

## Rebuild the Projection

The exporter uses the original join and validation functions. Given the
construction snapshot with `difficulty_score` and `difficulty_components`, and
the initial evaluations, repaired evaluations, prompts, and repaired semantic
replay outputs in the documented `runs/` layout:

```bash
python experiments/difficulty_validation/export_inputs.py \
  --benchmark /path/to/construction_snapshot.jsonl \
  --runs-dir /path/to/runs
```

The exporter verifies all 600 contract IDs per model and checks cumulative
stage totals against the published results. It does not reconstruct complexity
scores from decompiler performance.
