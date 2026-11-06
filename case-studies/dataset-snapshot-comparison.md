# When identical dataset files change label meaning

I built a small offline lab for [my keyed dataset-comparison proposal](https://github.com/huggingface/datasets/discussions/8906). It compares two complete local snapshots through the public `Dataset`, `Features` and `ClassLabel` APIs from Hugging Face Datasets 5.1.0.

The central fixture has two rows:

| Key | Text | Encoded label |
| --- | --- | --- |
| `r1` | `alpha` | 0 |
| `r2` | `beta` | 1 |

In the first snapshot, `ClassLabel.names` is `["negative", "positive"]`. In the second, it is `["positive", "negative"]`. The JSONL data bytes are identical, but both labels change meaning. Comparing the data-file digests or integer labels alone misses that difference. A metric calculated against the same encoded IDs could also stay unchanged; this example makes no model-quality claim.

I keep three observations separate: encoded row edits, decoded label meanings, and changes to the supplied features/configuration. I match rows by their exact, explicitly supplied string key. Reordering rows or repacking them into other shards can change file hashes while leaving the keyed comparison equal.

## Run it

From the repository root, use a fresh environment with Python 3.11 or newer:

```sh
python -m venv .venv-dataset-diff
```

Activate it with `source .venv-dataset-diff/bin/activate` on Linux/macOS or `.\.venv-dataset-diff\Scripts\Activate.ps1` in PowerShell, then run:

```sh
python -m pip install -r requirements-dataset-diff.txt
python examples/dataset_diff.py examples/dataset-diff-fixtures/base examples/dataset-diff-fixtures/relabel
```

The report includes `data_files_equal: true`, `features_changed: true`, two decoded label changes, and distinct content-derived snapshot IDs. These IDs are SHA256 digests of local manifests. They are not Hub commit SHAs.

Other inspectable fixtures cover the inverse and related cases:

| Compare `base` against | Keyed status | What changes |
| --- | --- | --- |
| `base` | `equal` | Nothing |
| `relabel` | `different` | Features and decoded meanings; identical data bytes |
| `config-only` | `different` | Explicit configuration name; identical data bytes |
| `repacked` | `equal` | Shard names, order and bytes; same keyed rows and declarations |
| `row-edits` | `different` | Add `r3`, remove `r2`, edit `r1` text and label |

Here, `equal` means equal within this declared comparison contract. It does not mean byte-identical artifacts. The report retains both snapshot IDs and all verified file digests.

## Input contract

Each snapshot contains a `manifest.json`, `features.json`, `config.json`, and zero or more JSONL shards. The manifest seals every declared file with its actual SHA256 and records the format version, key column, declared column order, and shard paths. Changing features, configuration, order or file bytes changes the complete snapshot identity.

I deliberately restrict the first slice to three flat columns: `id` and `text` as `Value("string")`, and `label` as `ClassLabel` with explicit names. Configuration is only the supplied `name` and `split`. Nested/media features, generic scalar types, dataset scripts and a Hub loader are outside this implementation.

The loader validates every raw row before native construction can coerce it: exact fields, nonempty string IDs, global uniqueness across shards, string text, and integer label IDs excluding booleans. It accepts `-1` as a distinct unknown-label sentinel; a real class named `unknown` remains a known class. It preserves whitespace and Unicode in keys without normalization.

I compare the feature mapping separately from the declared column order. Construction uses `Dataset.from_dict` in that declared order; I do not infer the source declaration from a normalized Dataset object.

The default budget is 1,000 rows and 1 MiB per snapshot, including manifest and metadata bytes. A budget overflow, digest mismatch, duplicate JSON member, unsupported declaration or invalid key fails the comparison. The CLI returns exit code 2 with `complete: false` for invalid input, and exit code 3 for an unexpected construction/runtime failure. A valid complete comparison returns 0 for either `equal` or `different`; callers must inspect `status` when using it as a gate. Increasing budgets permits a larger full scan, without adding streaming or digest shortcuts.

## Checks and limits

Run the independent oracles, CLI fixtures and isolated negative controls with:

```sh
python examples/dataset_diff_validation.py --output .lab/dataset-diff/results.json
```

The runner records exact dependency versions, test outcomes and CLI reports. Its negative controls deliberately ignore metadata when data hashes match, match rows by position, omit decoded meanings, overwrite duplicate keys, and certify a capped prefix. Each must cause a specific independent assertion to fail. The pristine implementation must pass the complete test file.

The [CI workflow](../.github/workflows/dataset-diff.yml) runs Linux Python 3.11/3.14 and Windows Python 3.12.10 with Hub access disabled. This is an independent educational example using synthetic data. It does not test remote Hub revision loading, authentication, concurrent file writers, large-dataset performance, model behavior, or downstream adoption. I have not delivered it as a workshop.

I ran the following matrix on 6 November 2026 at source `8fa50a94f15d1168ac6a340a5d2f3d9edf13ea0b`:

| Environment | Independent tests | Successful subtests | CLI scenarios | Broken controls caught |
| --- | --- | --- | --- | --- |
| Linux, Python 3.11.18 | 17 | 196 | 7 | 5 |
| Linux, Python 3.14.10 | 17 | 196 | 7 | 5 |
| Windows, Python 3.12.10 | 17 | 196 | 7 | 5 |

There were no skips, errors or failures in the pristine test suite. Every broken control failed its intended test assertion rather than an import/setup error. Ruff and the local-source mypy check also passed; that mypy invocation skips traversal of dependency internals. The [measured JSON reports and exact CI manifest](../validation/dataset-diff/manifest.json) retain the commands, package versions, expected-failure logs and full comparison reports. All 16 jobs for the repository passed at that source, including the existing JAX and ADK labs.

## Source references

I used the released [Datasets 5.1.0 source](https://github.com/huggingface/datasets/tree/efe6ea4f88260b1a5066f28a9c90d82365a29f42). Its [ClassLabel implementation](https://github.com/huggingface/datasets/blob/efe6ea4f88260b1a5066f28a9c90d82365a29f42/src/datasets/features/features.py) defines the encoded-ID/name mapping and rejects `int2str(-1)`. The [Dataset construction code](https://github.com/huggingface/datasets/blob/efe6ea4f88260b1a5066f28a9c90d82365a29f42/src/datasets/arrow_dataset.py) is the reason I validate raw records and retain declarations separately before construction.
