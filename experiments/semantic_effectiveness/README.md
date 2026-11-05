# Semantic Checkpoint Effectiveness and Coverage Sensitivity

This analysis reproduces the paper's checkpoint-effectiveness and
coverage-sensitivity tables for the original sample and repaired model outputs.

## Run Offline

From the repository root, using Python 3.10 or newer:

```bash
python experiments/semantic_effectiveness/analyze.py
python experiments/semantic_effectiveness/test_analyze.py
python experiments/semantic_effectiveness/check_results.py
```

Only the Python standard library is required. The command writes `results.json`
and `results.md` beside the script. No API key, compiler, or archive node is
needed to recompute these analyses from the included observations.

The results include 344,055 mismatches among 564,261 replayed model/checkpoint
comparisons and 22,837 failing model/callable-entry pairs. The analysis also
computes test-family contributions, mismatch modalities, and repaired semantic
consistency at each reported coverage threshold.

## Compact Inputs

The `inputs/` directory contains:

- `benchmark.jsonl.gz`: case IDs, callable signatures, test families, mutability,
  complexity strata, and archived per-contract coverage measurements.
- Four compressed replay files: each retains case identity, observed match
  status, the differing observables, and success/revert status.
- `manifest.json`: SHA-256 hashes of the original inputs and public projections.

These projections avoid duplicating contract source, bytecode, full return
values, logs, storage values, and execution logs. They preserve the per-case
observations needed for this analysis rather than just its final totals.
`export_inputs.py` derives differing observables from the original expected and
actual replay observations and verifies each stored match flag. Storage
differences are checked for state-changing functions, not `view` or `pure`
functions. The original aggregation and subset-selection functions are retained.

Coverage fields are the archived measurements used to select the reported
subsets. This command does not rerun coverage or assert that these measurements
come exclusively from the packaged checkpoints. The metadata preserves the
historical coverage backend when recorded. It is not a new uniform-coverage
measurement.

Effectiveness counts include only entries actually replayed. Coverage-filtered
semantic scores include all benchmark entries in the retained contracts,
assigning zero credit to missing or unrecovered entries. Entries from different
models are separate observations.

## Rebuild the Projections

Given the original construction snapshot with `semantic_coverage` and the full
repaired replay outputs:

```bash
python experiments/semantic_effectiveness/export_inputs.py \
  --benchmark /path/to/construction_snapshot.jsonl \
  --replay-dir /path/to/runs/semantic
```

The public benchmark JSONL supplies the checkpoints, while archived coverage is
provided in this analysis bundle because it is not part of the minimal public
dataset schema. Full semantic replay can be regenerated using the repository's
evaluation scripts and original model outputs. The offline table-reproduction
command above requires only this compact bundle.

`expected_results.json` contains the archived reference analysis. The result
checker excludes changed file paths and hashes but compares all scientific
results, including coverage distributions and threshold scores.
