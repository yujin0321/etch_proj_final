# Final Pipeline Wiring Design

## Goal

Connect the runtime pipeline to the final dataset, final model artifacts, and etch fault prompt library.

## Scope

- Use `data/train_v6_80.csv` for final retraining.
- Use `data/test_v6_20.csv` for local WebSocket simulation.
- Use `modeling_final.py` as the server inference engine.
- Load runtime artifacts from `models_final/`.
- Route known fault labels through `etch_fault_prompts.py` for engineer-facing reports.

## Design

`server.py` owns runtime wiring. It will import `InferenceEngine` from `modeling_final`, instantiate it with `model_dir="models_final"`, and stream local simulation rows from `data/test_v6_20.csv`. The CSV path and model directory will be module constants so tests and future operators can verify the active assets without reading function bodies.

`modeling_final.py` will support the improved autoencoder architecture used by `re_train_models_final.py`, while retaining the current basic autoencoder as a fallback. This keeps the user-requested file as the runtime engine and prevents `models_final/autoencoder.pth` load failures.

`shap_analysis.py` will default to `models_final/sensor_stats.json`. `re_train_models_final.py` will emit that file from normal training rows so SHAP output can show normal ranges for the same model family.

`agents/shap_agent.py` will call `etch_fault_prompts.build_prompt()` for known fault labels and append SHAP evidence to the user prompt. Unknown labels will keep the current generic SHAP prompt path.

## Validation

- Unit tests verify runtime constants and prompt routing.
- Import tests verify `modeling_final.InferenceEngine` exposes the final model defaults.
- A smoke command verifies the final engine can load `models_final`.
