# Changelog

## 2026-02-25

- `30_delta_cone_signature_diagnose.py` now falls back to `iter` when `step` is missing, with a warning, to support embeddings that use `iter` as the step column.
