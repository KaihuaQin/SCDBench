# No-External-Signature-Hints Ablation

The same 60 submitted-benchmark contracts are used in every row. The no-hint intervention removes only external selector/topic-to-text mappings. Selectors, event topics, assembly, checkpoints, and all other inputs remain unchanged.

| Setting | Format | Compile | ABI P/R/F1 | Semantic | Perfect |
|---|---:|---:|---:|---:|---:|
| With hints, initial | 55/60 (91.7%) | 40/60 (66.7%) | 0.912/0.579/0.709 | 329/1308 (25.2%) | 3/60 |
| Without hints, initial | 58/60 (96.7%) | 48/60 (80.0%) | 0.445/0.355/0.395 | 177/1308 (13.5%) | 2/60 |
| With hints, repaired | 55/60 (91.7%) | 51/60 (85.0%) | 0.912/0.756/0.827 | 408/1308 (31.2%) | 3/60 |
| Without hints, repaired | 58/60 (96.7%) | 55/60 (91.7%) | 0.455/0.407/0.430 | 203/1308 (15.5%) | 2/60 |

All metrics are cumulative over the original 60 contracts and their callable entries. Repaired rows preserve the initial format-stage result, matching the paper protocol.

## No-Hint Results by Complexity Stratum

| Stage | Difficulty | Format | Compile | ABI F1 | Semantic |
|---|---|---:|---:|---:|---:|
| Initial | Easy | 19/20 | 18/20 | 0.391 | 37/199 (18.6%) |
| Initial | Medium | 19/20 | 17/20 | 0.527 | 75/430 (17.4%) |
| Initial | Hard | 20/20 | 13/20 | 0.304 | 65/679 (9.6%) |
| Repaired | Easy | 19/20 | 19/20 | 0.390 | 37/199 (18.6%) |
| Repaired | Medium | 19/20 | 19/20 | 0.533 | 86/430 (20.0%) |
| Repaired | Hard | 20/20 | 17/20 | 0.370 | 80/679 (11.8%) |

## Main Result

Removing external mappings changes cumulative ABI F1 by -0.314 before repair and -0.397 after repair. Semantic consistency changes by -11.6 and -15.7 percentage points, respectively.

The no-hint arm cost $20.1844 for 60 initial responses and $0.6320 for 10 repairs.

## Interpretation Limit

The no-hint calls were run later against the undated model alias, while the stored with-hint calls report a dated snapshot and include two selected contracts without successful API responses. The large ABI difference is useful sensitivity evidence, but this historical-control comparison is not a clean contemporaneous causal estimate.
