# Difficulty Validation

## Scope

This analysis tests whether the continuous SCDBench difficulty score is associated with cumulative model outcomes. Format is measured on the original output; later stages use the artifact after allowed repair. It is an association analysis, not evidence that the score is a causal or complete measure of LLM decompilation hardness.

All primary outcomes are cumulative. An earlier-stage failure receives zero at later stages, matching the paper's fixed-denominator protocol.

## Data And Outcome Checks

| Model | Format | Compile | Exact ABI | Semantic equivalent | Semantic perfect |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opus 4.7 | 435/600 | 415/600 | 236/600 | 4284/14553 (29.4%) | 42/600 |
| GPT-5.3-Codex | 564/600 | 542/600 | 389/600 | 4078/14553 (28.0%) | 41/600 |
| GLM-5 | 520/600 | 403/600 | 174/600 | 3015/14553 (20.7%) | 18/600 |
| GLM-5 (instruct) | 540/600 | 386/600 | 122/600 | 2570/14553 (17.7%) | 6/600 |

Outcome definitions:

- `format_pass`: original decompilation output passes the format protocol.
- `compile_pass`: artifact after the allowed repair compiles.
- `abi_f1_cumulative`: evaluator per-contract ABI F1; zero if ABI evaluation is unavailable because a prerequisite failed.
- `semantic_rate_cumulative`: equivalent ground-truth functions divided by all benchmark semantic entries for that contract, including fallback/receive entries in the denominator; missing entries score zero.
- Binary robustness outcomes are exact ABI recovery and perfect contract-level semantic recovery.

## Unadjusted Spearman Associations

Negative rho means that higher difficulty is associated with worse performance. Confidence intervals resample contracts.

| Outcome | Model | rho | Bootstrap 95% CI | BH q |
| --- | --- | ---: | ---: | ---: |
| Format pass | Opus 4.7 | -0.217 | [-0.295, -0.135] | <0.001 |
| Format pass | GPT-5.3-Codex | -0.092 | [-0.162, -0.018] | 0.027 |
| Format pass | GLM-5 | 0.022 | [-0.061, 0.105] | 0.585 |
| Format pass | GLM-5 (instruct) | -0.026 | [-0.104, 0.052] | 0.555 |
| Format pass | Across-model mean | -0.183 | [-0.259, -0.102] | <0.001 |
| Compilation pass | Opus 4.7 | -0.261 | [-0.334, -0.183] | <0.001 |
| Compilation pass | GPT-5.3-Codex | -0.126 | [-0.196, -0.052] | 0.002 |
| Compilation pass | GLM-5 | -0.182 | [-0.261, -0.102] | <0.001 |
| Compilation pass | GLM-5 (instruct) | -0.232 | [-0.311, -0.156] | <0.001 |
| Compilation pass | Across-model mean | -0.358 | [-0.430, -0.287] | <0.001 |
| ABI F1 (cumulative) | Opus 4.7 | -0.322 | [-0.397, -0.246] | <0.001 |
| ABI F1 (cumulative) | GPT-5.3-Codex | -0.213 | [-0.291, -0.138] | <0.001 |
| ABI F1 (cumulative) | GLM-5 | -0.291 | [-0.363, -0.213] | <0.001 |
| ABI F1 (cumulative) | GLM-5 (instruct) | -0.294 | [-0.367, -0.217] | <0.001 |
| ABI F1 (cumulative) | Across-model mean | -0.377 | [-0.445, -0.302] | <0.001 |
| Semantic rate (cumulative) | Opus 4.7 | -0.227 | [-0.303, -0.151] | <0.001 |
| Semantic rate (cumulative) | GPT-5.3-Codex | -0.155 | [-0.236, -0.075] | <0.001 |
| Semantic rate (cumulative) | GLM-5 | -0.240 | [-0.317, -0.160] | <0.001 |
| Semantic rate (cumulative) | GLM-5 (instruct) | -0.218 | [-0.299, -0.137] | <0.001 |
| Semantic rate (cumulative) | Across-model mean | -0.327 | [-0.398, -0.248] | <0.001 |

## Descriptive Difficulty-Component Associations

The five component values are the frozen benchmark-construction scores in `[0, 1]`. These results are descriptive, unadjusted associations, not causal effects or independent component importance.

Each contract contributes one observation: its cumulative outcome averaged across the four models. Negative rho means a larger component is associated with worse performance. Confidence intervals resample complete contracts, and BH q-values use one family of 20 component-by-outcome tests.

| Component | Min | Median | Max | Unique | Zeros |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bytecode scale | 0 | 0.632 | 1.000 | 557 | 26 |
| Control flow | 0 | 0.602 | 1.000 | 567 | 22 |
| Interface/source | 0 | 0.582 | 1.000 | 567 | 15 |
| State interaction | 0 | 0.469 | 1.000 | 570 | 4 |
| Low-level Solidity | 0 | 0.300 | 0.900 | 16 | 150 |

| Component | Outcome | rho | Contract-bootstrap 95% CI | BH q (20 tests) |
| --- | --- | ---: | ---: | ---: |
| Bytecode scale | Format pass | -0.173 | [-0.247, -0.091] | <0.001 |
| Bytecode scale | Compilation pass | -0.328 | [-0.394, -0.255] | <0.001 |
| Bytecode scale | ABI F1 (cumulative) | -0.346 | [-0.412, -0.267] | <0.001 |
| Bytecode scale | Semantic rate (cumulative) | -0.291 | [-0.363, -0.211] | <0.001 |
| Control flow | Format pass | -0.164 | [-0.243, -0.085] | <0.001 |
| Control flow | Compilation pass | -0.326 | [-0.396, -0.255] | <0.001 |
| Control flow | ABI F1 (cumulative) | -0.349 | [-0.416, -0.274] | <0.001 |
| Control flow | Semantic rate (cumulative) | -0.297 | [-0.372, -0.213] | <0.001 |
| Interface/source | Format pass | -0.209 | [-0.284, -0.128] | <0.001 |
| Interface/source | Compilation pass | -0.370 | [-0.441, -0.300] | <0.001 |
| Interface/source | ABI F1 (cumulative) | -0.381 | [-0.450, -0.310] | <0.001 |
| Interface/source | Semantic rate (cumulative) | -0.315 | [-0.389, -0.237] | <0.001 |
| State interaction | Format pass | -0.151 | [-0.224, -0.067] | <0.001 |
| State interaction | Compilation pass | -0.234 | [-0.317, -0.153] | <0.001 |
| State interaction | ABI F1 (cumulative) | -0.271 | [-0.348, -0.186] | <0.001 |
| State interaction | Semantic rate (cumulative) | -0.205 | [-0.285, -0.124] | <0.001 |
| Low-level Solidity | Format pass | -0.051 | [-0.128, 0.026] | 0.208 |
| Low-level Solidity | Compilation pass | -0.177 | [-0.254, -0.096] | <0.001 |
| Low-level Solidity | ABI F1 (cumulative) | -0.161 | [-0.241, -0.081] | <0.001 |
| Low-level Solidity | Semantic rate (cumulative) | -0.198 | [-0.270, -0.113] | <0.001 |

## Adjusted Model-Fixed-Effect Results

The table reports the change in outcome percentage points for a one-standard-deviation increase in difficulty, controlling for `log1p` prompt characters, `log1p` public-function count, and model fixed effects. Standard errors are clustered by contract.

| Outcome | Difficulty effect (pp) | Cluster bootstrap 95% CI | BH q | R2 |
| --- | ---: | ---: | ---: | ---: |
| Format pass | -2.161 | [-5.956, 1.434] | 0.291 | 0.062 |
| Compilation pass | -9.588 | [-14.846, -4.723] | <0.001 | 0.093 |
| ABI F1 (cumulative) | -11.861 | [-17.870, -6.503] | <0.001 | 0.118 |
| Semantic rate (cumulative) | -8.876 | [-13.393, -4.439] | <0.001 | 0.096 |

## Binary Robustness Outcomes

| Outcome | Difficulty effect (pp) | Cluster bootstrap 95% CI | BH q | R2 |
| --- | ---: | ---: | ---: | ---: |
| Exact ABI recovery | -11.568 | [-18.292, -4.878] | <0.001 | 0.215 |
| Perfect semantic recovery | -0.615 | [-4.117, 2.890] | 0.743 | 0.113 |

## Predictor Diagnostics

| Predictor | VIF |
| --- | ---: |
| `difficulty_score` | 7.718 |
| `log1p_prompt_char_length` | 8.045 |
| `log1p_public_function_count` | 2.231 |

Pearson predictor correlation matrix:

|  | Difficulty | Log prompt chars | Log public functions |
| --- | ---: | ---: | ---: |
| Difficulty | 1.000 | 0.931 | 0.722 |
| Log prompt chars | 0.931 | 1.000 | 0.736 |
| Log public functions | 0.722 | 0.736 | 1.000 |

## Interpretation

Higher difficulty has a negative adjusted coefficient for 4/4 primary outcomes; 3/4 contract-bootstrap intervals lie fully below zero.

The adjusted coefficient is the association with the residual part of the composite difficulty score after controlling for two components closely related to that score. Predictor correlations and VIFs should therefore be considered when interpreting coefficient magnitude.

OLS is used deliberately to estimate an average score/probability difference with contract-clustered inference. It does not assert that predictions outside [0, 1] are meaningful, and no causal interpretation is intended.

## Reproduction

```bash
python experiments/difficulty_validation/analyze.py --bootstrap-replicates 2000 --seed 20260724
```
