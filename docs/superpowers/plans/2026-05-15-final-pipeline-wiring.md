# Final Pipeline Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the application runtime to final data, final model artifacts, and the new etch fault prompt library.

**Architecture:** Keep runtime wiring in `server.py`, model compatibility in `modeling_final.py`, stats emission in `re_train_models_final.py`, and LLM prompt routing in `agents/shap_agent.py`. Add small tests that verify the active contracts without booting the full FastAPI lifespan.

**Tech Stack:** Python, FastAPI, PyTorch, LightGBM, SHAP, LangChain, pytest.

---

### Task 1: Runtime Constants And Final Engine

**Files:**
- Modify: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/server.py`
- Modify: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/modeling_final.py`
- Test: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/tests/test_final_pipeline_wiring.py`

- [ ] Add failing tests asserting `server.LOCAL_SIMULATION_CSV == "data/test_v6_20.csv"`, `server.MODEL_DIR == "models_final"`, and `modeling_final.InferenceEngine().model_dir` defaults to `models_final`.
- [ ] Run `python3 -m pytest tests/test_final_pipeline_wiring.py -q` and confirm failure because constants/defaults are missing or stale.
- [ ] Add constants to `server.py`, import `InferenceEngine` from `modeling_final`, and instantiate with `MODEL_DIR`.
- [ ] Change `modeling_final.InferenceEngine` default `model_dir` to `models_final`.
- [ ] Add `ImprovedAutoencoder` support in `modeling_final.py` and auto-select it when saved weights match that architecture.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Final CSV Simulation Compatibility

**Files:**
- Modify: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/server.py`
- Test: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/tests/test_final_pipeline_wiring.py`

- [ ] Add a failing test proving local simulation uses `LOCAL_SIMULATION_CSV`.
- [ ] Run the focused test and confirm failure.
- [ ] Replace hard-coded `data/test_tstr.csv` with `LOCAL_SIMULATION_CSV`.
- [ ] Use `Run_Name` when `run_id` is absent so `test_v6_20.csv` streams cleanly.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Prompt Routing

**Files:**
- Modify: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/agents/shap_agent.py`
- Test: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/tests/test_final_pipeline_wiring.py`

- [ ] Add a failing test that monkeypatches `ChatOpenAI` and verifies known faults route through `etch_fault_prompts`.
- [ ] Run the focused test and confirm failure.
- [ ] Update `SHAPAgent` to build known-fault prompts with `etch_fault_prompts.build_prompt()` and append SHAP evidence.
- [ ] Preserve generic fallback behavior for unknown labels.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Final Sensor Stats

**Files:**
- Modify: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/re_train_models_final.py`
- Modify: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/shap_analysis.py`
- Test: `/Users/ichanmin/Desktop/git_prac/etch_proj_final/tests/test_final_pipeline_wiring.py`

- [ ] Add failing tests asserting SHAP defaults to `models_final/sensor_stats.json` and retraining exposes a stats builder.
- [ ] Run the focused tests and confirm failure.
- [ ] Emit `models_final/sensor_stats.json` during retraining from normal rows.
- [ ] Change SHAP default stats path to `models_final/sensor_stats.json`.
- [ ] Run the focused tests and confirm they pass.

### Task 5: Verification

**Files:**
- Runtime verification only.

- [ ] Run `python3 -m pytest tests/test_final_pipeline_wiring.py -q`.
- [ ] Run `python3 - <<'PY'` smoke import/load for `modeling_final.InferenceEngine(model_dir="models_final")`.
- [ ] Run `git diff --stat` and inspect changed files.
