# ItyFuzz Coverage Comparison

This directory contains SCDBench's invocation adapter, coverage analysis, and
recorded coverage measurements.

ItyFuzz is an external dependency:

- Repository: https://github.com/fuzzland/ityfuzz
- Revision: `80e3b44170573b592cd53b2c5ee9f36b21032b16`

The experiment uses the original SCDBench sample. `selected_ids.txt` contains
20 easy, 20 medium, and 20 hard contracts, matching the signature-hint subset.

## Reproduce the Reported Coverage Table Offline

Run from the repository root with Python 3.10 or newer:

```bash
python experiments/ityfuzz_comparison/summarize_matched_coverage.py
```

The script recomputes source-line intersections and aggregation from the
recorded inputs. It requires neither ItyFuzz nor an archive node.
The output is `experiments/ityfuzz_comparison/results/matched_coverage.json`.

| Method | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| SCDBench | 88.8% | 45.7% | 72.1% | 67.9% |
| ItyFuzz | 30.2% | 25.7% | 20.6% | 24.0% |

These results use 51 contracts with comparable coverage measurements
(20 easy, 15 medium, and 16 hard). Unavailable measurements are not assigned
zero. Default ItyFuzz does not initialize fuzzing for `view`-only targets.
Three such targets and one local deployment failure have zero coverage.
Three runs retain partial coverage persisted before termination.

## Scope and Timing

The ItyFuzz search budgets are 4, 9, and 16 seconds for easy, medium, and hard
contracts. They were selected using SCDBench checkpoint-replay timings.
SCDBench timing excludes checkpoint generation, compilation, and coverage
instrumentation. ItyFuzz coverage replay is also performed separately from its
search budget. This is therefore not an equal-total-cost comparison of two
test generators.

Four SCDBench over-budget cases have recorded deadline-prefix reruns in
`recorded/budgeted_prefix/`. The original full-suite coverage remains in use for
`scdbench-medium-163`, whose measured replay took about 123.6 seconds. No recorded
9-second prefix is available for that contract. The table must not be described
as imposing the same hard deadline on every SCDBench run.

This comparison measures coverage, not vulnerability discovery or differential
semantic mismatches. SCDBench sequences contain at most two calls. ItyFuzz's
recorded maximum exported sequence depth is 21. Only sequence counts and depths
are included, not transaction traces or vulnerability outputs.

## Recorded Inputs

- `recorded/contract_metadata.jsonl`: selected contracts' archived coverage line
  sets and deployment modes, omitted from the compact public benchmark file.
- `recorded/ityfuzz_results.json`: coverage line sets, run statuses, and budgets.
- `recorded/budgeted_prefix/`: the four measured SCDBench prefix coverage records.
- `recorded/provenance.json`: hashes of the source artifacts and projection scope.

`export_recorded_inputs.py` documents the allowlisted projection from the
archived dataset and experiment records. It preserves measurements while
omitting process logs, private paths, RPC endpoints, and vulnerability output.

## Run New Local Campaigns

Download the original `scdbench.jsonl` from
https://doi.org/10.7910/DVN/IPPTIX. Install the repository's Python requirements,
Foundry, and the build prerequisites documented by ItyFuzz. Keep the upstream
checkout outside this repository:

```bash
git clone https://github.com/fuzzland/ityfuzz.git ../ityfuzz
git -C ../ityfuzz checkout 80e3b44170573b592cd53b2c5ee9f36b21032b16
cargo build --release --manifest-path ../ityfuzz/Cargo.toml

python experiments/ityfuzz_comparison/prepare_artifacts.py \
  --dataset ../dataset/scdbench.jsonl \
  --ids-file experiments/ityfuzz_comparison/selected_ids.txt \
  --output-dir experiments/ityfuzz_comparison/generated

python experiments/ityfuzz_comparison/run_ityfuzz.py \
  --prepared-manifest experiments/ityfuzz_comparison/generated/prepared_manifest.json \
  --ityfuzz-repo ../ityfuzz \
  --output-dir experiments/ityfuzz_comparison/runs \
  --wall-seconds-by-difficulty easy=4,medium=9,hard=16

python experiments/ityfuzz_comparison/summarize_matched_coverage.py \
  --ityfuzz-results experiments/ityfuzz_comparison/runs/results.json \
  --output experiments/ityfuzz_comparison/results/new_matched_coverage.json
```

The adapter checks exact compiled runtime-bytecode agreement and reproduces the
recorded deployment mode. It runs only locally supplied artifacts, disables
detectors, and verifies the upstream commit. Campaigns run sequentially.
Wall-clock-bounded runs can vary with hardware and toolchain versions.

Tests:

```bash
python -m unittest discover -s experiments/ityfuzz_comparison -p 'test_*.py' -v
```
