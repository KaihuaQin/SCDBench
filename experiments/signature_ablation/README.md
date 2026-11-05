# Signature-Hint Ablation

This experiment evaluates GPT-5.3-Codex with high reasoning on 60 contracts
from the original sample, with 20 from each complexity stratum. The intervention
removes the external function-selector and event-signature candidate mappings.
Assembly, intrinsic selectors and topics, and semantic checkpoints are unchanged.

## Reproduce the Tables Offline

From the repository root, using Python 3.10 or newer:

```bash
python experiments/signature_ablation/analyze_results.py
python -m unittest discover -s experiments/signature_ablation -p 'test_*.py'
```

The analyzer writes `results/results.json` and `results/RESULTS.md`. It recomputes
all initial and repaired metrics from the per-contract inputs, rather than
copying a stored summary. No API key, Ethereum node, compiler, or paid model call
is needed for this analysis. Use `--input-dir` and output arguments to override
the default locations.

## Included Inputs

- `data/subset_metadata.jsonl` identifies all 60 contracts and their callable
  signatures and denominator counts. It is not a replacement for the full dataset.
- `data/*_eval.jsonl` contains the fields consumed by the metric calculations,
  projected from the archived format, compilation, and ABI evaluations.
- `data/*_semantic.json` contains per-function replay results and totals.
- `data/*_targets.json` retains target IDs for validation, not target bytecode.
- `runs/` contains sanitized initial and repair responses for the no-hint arm
  and the paired with-hint control. Content, model settings, usage, and outcome
  fields are retained. Private paths, response identifiers, timestamps, and
  encrypted reasoning payloads are omitted, matching the original public release.
- `data/provenance.json` records the source-file names and checksums.
- `data/expected_results.json` is the archived result used only by regression tests.

The analysis preserves the initial format result after repair, as in the paper.
Missing or unrecovered entries remain in the cumulative denominators. The
with-hint responses are the stored original runs. They were not regenerated
contemporaneously with the no-hint calls.

## Recreate the Paired Datasets

Download the original `scdbench.jsonl` from https://doi.org/10.7910/DVN/IPPTIX,
then run:

```bash
python experiments/signature_ablation/prepare_subset.py --dataset ../dataset/scdbench.jsonl
```

This writes `generated/with_hints.jsonl` and `generated/without_hints.jsonl` for
the exact released subset. Use the public inference, repair, and evaluation
scripts with these files and the stored predictions. Replaying the actual
checkpoints requires the Foundry and archive-node setup in the root README.

`export_inputs.py` documents and performs the projections from the archived
experiment files. Its two source-directory arguments are explicit, with no
private filesystem defaults. `metrics.py` preserves the original sampling-
stability analysis's metric aggregation functions.
