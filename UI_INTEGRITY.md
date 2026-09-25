# UI integrity contract

This repository is allowed to change **presentation**, not TWM dynamics.

## Frozen model boundary

The following files are copied from `codxqqq-lab/twm@main` and must remain byte-identical unless a new canonical architecture is explicitly promoted:

- `runtime/canonical/factor_model.py`
- `runtime/canonical/separated_system_id.py`
- `runtime/canonical/distilled_evidence_system_id.py`
- `runtime/engine.py`
- `runtime/schema.py`
- `runtime/ui_adapter.py`
- `examples/station_7.json`
- `examples/station_8.json`
- `examples/station_9.json`
- `examples/station_10.json`

Their canonical SHA-256 values live in `SOURCE_PROVENANCE.json` where applicable.

## Allowed UI transformations

- rename node 0 → “Модуль A” for display;
- cards, graph rendering, prose explanations;
- group/hide technical fields;
- derive human-readable counts from neural outputs;
- derive an explicitly labelled *belief concentration* visualization.

These transformations never feed back into inference.

## Recorded future boundary

`scenario["recorded"]` may be consulted only **after** a neural prediction exists, through
`runtime.schema.observed_next`, to show a factual comparison. It must never be included in
the input to `predict_streamlit_payload`.

## Weight bootstrap

`runtime/weight_bootstrap.py` downloads canonical public checkpoints from
`codxqqq-lab/twm` and validates SHA-256 before they can be used. `runtime.engine`
then validates SHA again. This is deployment plumbing only.
