# Semantic-Test Effectiveness

This analysis uses compact inputs exported from the submitted benchmark snapshot and repaired semantic replay artifacts. Testcase-effectiveness denominators include only functions actually replayed for each model. Coverage-threshold rankings instead use the cumulative benchmark denominator, so missing or unrecovered functions count as failures.

## Snapshot Checks

- Contracts: **600**
- Executable ABI entry groups: **14,553**
- Concrete semantic cases: **227,383**
- Benchmark SHA-256: `ace3bc28a175887f1b735eebb4c80ff6c62aa5898d2217ecf8e523f90cb9ba44`

## Case Composition

| Category | Cases | Share | ABI entry groups | Contracts |
| --- | --- | --- | --- | --- |
| random | 91,454 | 40.2% | 14,553 | 600 |
| boundary | 32,304 | 14.2% | 8,175 | 586 |
| static | 77,069 | 33.9% | 7,153 | 572 |
| stateful | 26,556 | 11.7% | 13,754 | 524 |

The four category counts sum to 227,383. `boundary` combines all `branch_boundary:*` variants, and `static` combines all `static_hint:*` variants.

## Coverage

Coverage is marked available for **500/600** contracts. Line coverage is numerically measurable for **499**; one available record has no line ratio. Coverage is diagnostic and is not used to decide semantic equivalence.

| Metric | Measurable | Covered/total | Pooled | Macro mean | Median | P10 |
| --- | --- | --- | --- | --- | --- | --- |
| lines | 499 | 42,107/54,122 | 77.8% | 69.6% | 76.5% | 18.4% |
| functions | 497 | 5,526/6,638 | 83.2% | 84.6% | 100.0% | 46.9% |
| branches | 326 | 2,327/7,508 | 31.0% | 36.6% | 33.3% | 0.0% |

The paper's 77.8% figure is pooled target-contract line coverage (42,107/54,122), not the per-contract mean (69.6%).

Unavailable reasons: forge_coverage_failed: 43, forge_coverage_ir_minimum_timed_out_after_300s: 1, foundry_coverage_parser_failed: 29, legacy_solidity_0_4_coverage_skipped: 17, synthetic_constructor_arg_literal_unsupported:string[4]: 1, synthetic_constructor_arg_literal_unsupported:tuple: 1, target_contract_no_executable_lines: 8.

## Mismatch Detection by Case Category

### Claude Opus 4.7 + repair

Denominator: **8,498** model-function evaluations and **130,605** replayed cases; **4,214** functions have at least one detected mismatch.

| Category | Cases | Mismatches | Mismatch rate |
| --- | --- | --- | --- |
| random | 52,573 | 23,987 | 45.6% |
| boundary | 18,927 | 8,356 | 44.1% |
| static | 43,340 | 20,122 | 46.4% |
| stateful | 15,765 | 7,238 | 45.9% |

Function-level contribution:

| Category | Functions with cases | Detected failures | Exclusive/removal loss | Beyond random | 1/k credit |
| --- | --- | --- | --- | --- | --- |
| random | 8,498 | 3,906 | 72 | n/a | 1478.5 |
| boundary | 4,792 | 2,296 | 44 | 99 | 660.7 |
| static | 4,090 | 1,979 | 49 | 104 | 555.3 |
| stateful | 8,172 | 3,866 | 160 | 161 | 1519.5 |

Mismatch modalities overlap. Of **59,703** mismatched calls: status=24,935, returndata=49,999, logs=8,993, storage=15,432.

### GPT-5.3-Codex + repair

Denominator: **12,227** model-function evaluations and **187,666** replayed cases; **8,149** functions have at least one detected mismatch.

| Category | Cases | Mismatches | Mismatch rate |
| --- | --- | --- | --- |
| random | 74,635 | 47,478 | 63.6% |
| boundary | 26,852 | 17,050 | 63.5% |
| static | 63,288 | 42,174 | 66.6% |
| stateful | 22,891 | 14,429 | 63.0% |

Function-level contribution:

| Category | Functions with cases | Detected failures | Exclusive/removal loss | Beyond random | 1/k credit |
| --- | --- | --- | --- | --- | --- |
| random | 12,227 | 7,625 | 103 | n/a | 2829.7 |
| boundary | 6,789 | 4,540 | 85 | 176 | 1279.0 |
| static | 5,898 | 4,005 | 55 | 145 | 1077.0 |
| stateful | 11,852 | 7,661 | 293 | 295 | 2963.3 |

Mismatch modalities overlap. Of **121,131** mismatched calls: status=50,923, returndata=104,292, logs=16,392, storage=26,043.

### GLM-5 high + repair

Denominator: **8,302** model-function evaluations and **126,976** replayed cases; **5,287** functions have at least one detected mismatch.

| Category | Cases | Mismatches | Mismatch rate |
| --- | --- | --- | --- |
| random | 51,246 | 32,902 | 64.2% |
| boundary | 18,431 | 11,848 | 64.3% |
| static | 41,822 | 28,860 | 69.0% |
| stateful | 15,477 | 9,181 | 59.3% |

Function-level contribution:

| Category | Functions with cases | Detected failures | Exclusive/removal loss | Beyond random | 1/k credit |
| --- | --- | --- | --- | --- | --- |
| random | 8,302 | 4,903 | 92 | n/a | 1775.4 |
| boundary | 4,662 | 3,220 | 74 | 153 | 926.1 |
| static | 3,985 | 2,784 | 39 | 116 | 753.8 |
| stateful | 8,008 | 4,882 | 192 | 196 | 1831.8 |

Mismatch modalities overlap. Of **82,791** mismatched calls: status=33,507, returndata=69,428, logs=12,945, storage=22,760.

### GLM-5 instruct + repair

Denominator: **7,757** model-function evaluations and **119,014** replayed cases; **5,187** functions have at least one detected mismatch.

| Category | Cases | Mismatches | Mismatch rate |
| --- | --- | --- | --- |
| random | 47,720 | 31,953 | 67.0% |
| boundary | 17,185 | 11,327 | 65.9% |
| static | 39,632 | 28,144 | 71.0% |
| stateful | 14,477 | 9,006 | 62.2% |

Function-level contribution:

| Category | Functions with cases | Detected failures | Exclusive/removal loss | Beyond random | 1/k credit |
| --- | --- | --- | --- | --- | --- |
| random | 7,757 | 4,870 | 98 | n/a | 1785.4 |
| boundary | 4,346 | 3,099 | 55 | 114 | 879.4 |
| static | 3,768 | 2,681 | 33 | 86 | 719.8 |
| stateful | 7,501 | 4,791 | 170 | 176 | 1802.4 |

Mismatch modalities overlap. Of **80,430** mismatched calls: status=30,267, returndata=65,280, logs=12,339, storage=23,064.

## Exact Case-Kind Mismatch Rates

| Model | Exact case_kind | Cases | Mismatches | Mismatch rate |
| --- | --- | --- | --- | --- |
| Claude Opus 4.7 + repair | branch_boundary:moderate_threshold | 4,777 | 2,080 | 43.5% |
| Claude Opus 4.7 + repair | branch_boundary:one_true | 4,781 | 2,169 | 45.4% |
| Claude Opus 4.7 + repair | branch_boundary:small_threshold | 4,783 | 2,183 | 45.6% |
| Claude Opus 4.7 + repair | branch_boundary:zero_empty | 4,586 | 1,924 | 42.0% |
| Claude Opus 4.7 + repair | deterministic_random | 52,573 | 23,987 | 45.6% |
| Claude Opus 4.7 + repair | stateful_setup | 15,765 | 7,238 | 45.9% |
| Claude Opus 4.7 + repair | static_hint:literal | 43,340 | 20,122 | 46.4% |
| GPT-5.3-Codex + repair | branch_boundary:moderate_threshold | 6,773 | 4,252 | 62.8% |
| GPT-5.3-Codex + repair | branch_boundary:one_true | 6,771 | 4,331 | 64.0% |
| GPT-5.3-Codex + repair | branch_boundary:small_threshold | 6,775 | 4,364 | 64.4% |
| GPT-5.3-Codex + repair | branch_boundary:zero_empty | 6,533 | 4,103 | 62.8% |
| GPT-5.3-Codex + repair | deterministic_random | 74,635 | 47,478 | 63.6% |
| GPT-5.3-Codex + repair | stateful_setup | 22,891 | 14,429 | 63.0% |
| GPT-5.3-Codex + repair | static_hint:literal | 63,288 | 42,174 | 66.6% |
| GLM-5 high + repair | branch_boundary:moderate_threshold | 4,647 | 2,939 | 63.2% |
| GLM-5 high + repair | branch_boundary:one_true | 4,653 | 3,021 | 64.9% |
| GLM-5 high + repair | branch_boundary:small_threshold | 4,653 | 3,048 | 65.5% |
| GLM-5 high + repair | branch_boundary:zero_empty | 4,478 | 2,840 | 63.4% |
| GLM-5 high + repair | deterministic_random | 51,246 | 32,902 | 64.2% |
| GLM-5 high + repair | stateful_setup | 15,477 | 9,181 | 59.3% |
| GLM-5 high + repair | static_hint:literal | 41,822 | 28,860 | 69.0% |
| GLM-5 instruct + repair | branch_boundary:moderate_threshold | 4,332 | 2,772 | 64.0% |
| GLM-5 instruct + repair | branch_boundary:one_true | 4,335 | 2,929 | 67.6% |
| GLM-5 instruct + repair | branch_boundary:small_threshold | 4,336 | 2,976 | 68.6% |
| GLM-5 instruct + repair | branch_boundary:zero_empty | 4,182 | 2,650 | 63.4% |
| GLM-5 instruct + repair | deterministic_random | 47,720 | 31,953 | 67.0% |
| GLM-5 instruct + repair | stateful_setup | 14,477 | 9,006 | 62.2% |
| GLM-5 instruct + repair | static_hint:literal | 39,632 | 28,144 | 71.0% |

## Coverage-Threshold Ranking Sensitivity

Scores below use all submitted function groups in the selected contracts, including zero credit for missing or unrecovered functions.

| Subset | Contracts | Functions | Opus + repair | GPT + repair | GLM high + repair | GLM instruct + repair |
| --- | --- | --- | --- | --- | --- | --- |
| all_contracts | 600 | 14,553 | 29.4% | 28.0% | 20.7% | 17.7% |
| coverage_measurable | 499 | 11,732 | 30.4% | 28.5% | 19.3% | 16.9% |
| coverage_ge_25pct | 441 | 10,301 | 31.0% | 29.0% | 20.1% | 17.1% |
| coverage_ge_50pct | 369 | 8,812 | 30.5% | 27.9% | 19.9% | 17.5% |
| coverage_ge_75pct | 257 | 6,253 | 30.0% | 28.8% | 21.1% | 16.9% |
| coverage_ge_80pct | 239 | 5,770 | 29.7% | 29.5% | 21.4% | 17.2% |

The ordering is unchanged at every threshold, although Opus and GPT become nearly tied at the 80% threshold. This sensitivity analysis conditions on measurable source coverage and does not turn finite testing into a proof of semantic equivalence.

## Contribution Definitions

- **Detected failures:** replayed model-function evaluations with at least one mismatching case from the category.
- **Exclusive/removal loss:** failures detected by that category and no other; removing the category would make those functions pass all remaining cases.
- **Beyond random:** failures detected by the category where random cases found no mismatch; augmented categories can overlap.
- **1/k credit:** order-independent attribution. A failing function detected by `k` categories gives each detecting category `1/k` credit; credits sum to all detected failing functions.
