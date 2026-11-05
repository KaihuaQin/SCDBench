# External Validity

| Statistic | All sampled deployments | Unique deployed runtimes | Unique verified runtimes | SCDBench |
|---|---:|---:|---:|---:|
| Contracts | 1,000,000 | 37,653 | 772,736 | 600 |
| Median runtime bytes | 45 | 2,100 | 6,831 | 7,830 |
| Canonical EIP-1167 minimal proxy | 44.63% | 16.94% | 0.00% | 0.00% |
| Contains SLOAD | 19.54% | 64.45% | 96.41% | 93.50% |
| Contains SSTORE | 8.44% | 54.85% | 87.52% | 89.83% |
| Contains DELEGATECALL | 56.39% | 36.96% | 11.40% | 9.83% |

Unique deployed versus verified opcode total-variation distance: 0.035.

Compiler metadata is recoverable for 54.56% of unique deployed runtimes. Solidity 0.8 accounts for 75.46% of recovered versions and 77.45% of the verified corpus.

Deployed-population comparison, not a verification-labeled unverified-contract accuracy set.
