# External Validity

This analysis compares one million sampled deployed Ethereum contracts, their
37,653 distinct runtime bytecodes, the 772,736-contract source-verified corpus,
and the original 600-contract SCDBench sample. The deployed sample was collected
without verification labels. It is not a labeled unverified-contract dataset.

## Reproduce the Reported Table

From the repository root, using Python 3.10 or newer:

```bash
python experiments/external_validity/analyze.py
python -m unittest discover -s experiments/external_validity -p 'test_*.py'
```

The first command recomputes medians, low-level feature proportions, compiler
shares, and opcode-distribution distances. It writes `results/results.json` and
`results/REPORT.md`. These commands are offline and use only the standard library.

## Inputs and Weighting

`data/population_counts.json` contains numeric feature histograms, opcode counts,
compiler counts, and feature-presence counts. These are aggregated from the
archived bytecode and feature records, not the formatted paper table.

The deployment-weighted view counts each of the one million sampled addresses.
The unique-runtime view counts each exact runtime bytecode once. Weighted
histograms preserve the original quantile calculations without distributing
duplicate bytecode. The full verified opcode comparison uses its recorded top
80 opcode categories plus an OTHER category, matching the original analysis.

Source-file checksums and selection counts are included. Addresses, database
credentials, RPC endpoints, local paths, and individual bytecode blobs are not.
The full raw collection databases and verified corpus are not bundled here.
Consequently, the included inputs reproduce the reported statistical analysis,
but do not independently recollect or reconstruct the original address sample.

## Raw-Input Analysis

`from_bytecode.py` retains the original runtime scanner, compiler-metadata
decoder, population weighting, and comparison calculations. It uses the local
`collect_corpus_statistics.py` and `deduplicate_contract_dataset.py` helpers.
All raw-input paths can be supplied on the command line. Run
`python experiments/external_validity/from_bytecode.py --help` for its arguments.

`export_inputs.py` creates the compact input file from the archived SQLite
collections, original sample, and verified-corpus feature records. It uses the
same scanner and selection logic as `from_bytecode.py`. This export reads local
files only and makes no database-server or blockchain calls.

Compiler recovery is conditional on usable metadata. Low-level patterns and
opcode distributions describe the populations, not decompilation accuracy or
whether contracts appeared in model training data.
