# Sequential Semantic Checkpoint Timing

Completed: **60/60 contracts**.

| Bin | Contracts | Checkpoints | Checkpoint time | Median/contract | Submitted coverage | Measurable | Max depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 20/20 | 2613 | 35.31s | 1.59s | 89.0% | 20/20 | 2 |
| Medium | 20/20 | 6486 | 248.84s | 4.08s | 45.7% | 15/20 | 2 |
| Hard | 20/20 | 11290 | 149.35s | 7.13s | 74.8% | 17/20 | 2 |
| All | 60/60 | 20389 | 433.50s | 3.80s | 69.6% | 52/60 | 2 |

Checkpoint time excludes ground-truth compilation and coverage instrumentation. Submitted coverage is pooled executable-line coverage from the frozen benchmark artifact over contracts with numeric measurements. Fresh stored-cases-only LCOV is retained separately in results.json as a diagnostic.
