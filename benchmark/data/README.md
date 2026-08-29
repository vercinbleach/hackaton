# Benchmark suites

`google-founder-projection-dev.jsonl` is a development regression suite. Its examples and
semantics were visible during schema and threshold design, so its scores cannot be reported as
holdout performance.

`holdout-v1.jsonl` is a sealed, one-shot test. Do not generate predictions, tune thresholds, edit
the contract, or select a checkpoint after inspecting its results. Freeze the model revision,
schema, compiler policy, and thresholds first; verify the file hash against its manifest; then run
the full-plan scorer once.

Version 1 contains 40 manually reviewed cases and is a pilot holdout. Even 40 correct accepted
predictions would not establish 99% precision. A one-sided 95% exact bound needs at least 299
independent accepted cases with zero errors before a 99% claim is statistically supportable.

Raw generations belong under `benchmark/runs/` and are not evidence by themselves. A report must
state the suite, input hash, model revision, thresholds, coverage, accepted precision, exact-plan
accuracy, component accuracy, abstentions, and unsafe accepts.
