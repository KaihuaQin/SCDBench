# Semantic Checkpoint Timing

This directory contains SCDBench's sequential replay and deadline-prefix timing
scripts for the original 60-contract subset (20 per complexity stratum).
`selected_ids.txt` preserves the selected contracts.

## Reproduce Recorded Summaries Offline

From the repository root, using Python 3.10 or newer:

```bash
python experiments/semantic_checkpoint_timing/run_sequential.py \
  --summarize-only \
  --output-dir experiments/semantic_checkpoint_timing/results
```

This recomputes JSON, CSV, and Markdown summaries from
`recorded/timing_results.json`, without Foundry, an archive node, or model calls.
The inputs retain per-contract timings, checkpoint counts, baseline-validation
results, and the two separately labeled coverage measurements. Process logs and
private connection details are excluded.

All 60 original runs completed their stored checkpoints with matching baseline
observations. Compilation, checkpoint execution, and coverage instrumentation
times are recorded separately. Coverage in the timing summary is not the same
matched subset used in the ItyFuzz comparison. Use the neighboring
`ityfuzz_comparison/summarize_matched_coverage.py` for the paper's fuzzer table.

The 4/9/16-second fuzzer budgets were informed by these measurements, but are not
hard maxima of all original replay times. In particular,
`scdbench-medium-163` took about 123.6 seconds and has no archived budget-prefix
coverage rerun. The other four deadline-prefix coverage records are retained in
`../ityfuzz_comparison/recorded/budgeted_prefix/`.

## Run New Measurements

Download the original dataset from https://doi.org/10.7910/DVN/IPPTIX and install
the repository's Python requirements and Foundry. Set `ARCHIVE_RPC_URL` in your
environment to an archive endpoint that supports Ethereum block `24136053`.
The scripts do not include a node endpoint or credentials.

```bash
python experiments/semantic_checkpoint_timing/run_sequential.py \
  --dataset ../dataset/scdbench.jsonl \
  --output-dir experiments/semantic_checkpoint_timing/runs
```

The command recompiles ground truth, warms the Foundry harness once, interleaves
the complexity strata, and runs contracts and 10-checkpoint chunks sequentially.
Each replay is checked against its stored baseline. Dataset metadata needed for
the original deployment and coverage definitions is provided in
`../ityfuzz_comparison/recorded/contract_metadata.jsonl`.

The optional homogeneous LCOV diagnostic uses the frozen definition in
`recorded/coverage_protocol.json`. It is separate from archived coverage.
Coverage instrumentation time is excluded from the reported replay time.

Deadline-prefix reruns can be performed with:

```bash
python experiments/semantic_checkpoint_timing/run_budgeted_prefix.py \
  --dataset ../dataset/scdbench.jsonl \
  --output-dir experiments/semantic_checkpoint_timing/budgeted_prefix
```

That command targets all five recorded over-budget contracts, including the
medium contract for which no historical prefix result exists. A new result for
that contract is a new measurement, not part of the published table.

The scripts are resumable. Use `--force` only for intentional reruns. Machine
load, hardware, archive-node latency, and tool versions can change measured time.

Tests:

```bash
python -m unittest discover -s experiments/semantic_checkpoint_timing -p 'test_*.py' -v
```
