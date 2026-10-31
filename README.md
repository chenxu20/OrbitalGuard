# OrbitalGuard

**See tomorrow’s collision risks, today.** A functional Streamlit prototype for conjunction analysts: causal early-warning forecasts, deterministic proximity screening, held-out historical replay, and measured analyst-review comparisons. Decision support only; no flight commands or autonomous maneuvers.

## Run

Python 3.10+ supported. From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501. The supplied `data/train_data.csv` loads by default. The sidebar accepts another local file or an upload (400 MB cap). CSV/XLSX and a ZIP containing exactly one CSV are supported. Leading-zero event IDs survive ingestion. A local file avoids browser upload overhead. With no data or model, the app provides an honest empty state or exploration and distance screening.

This workspace’s precreated virtual environment inherits scientific packages because system `ensurepip` is absent. It already has the required app dependencies. Use `.venv/bin/python` to run the following commands.

## Reproduce training and verification

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/train.py --data data/train_data.csv --output artifacts/
.venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/smoke_test.py
```

The virtual environment includes the pinned pytest 8.4.2. On environments with unrelated globally installed pytest plugins, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` can isolate these project tests.

Training writes trusted local `model.joblib`, strict JSON metrics, validation/test prediction CSVs, cohort and split manifest into `artifacts/`. Generated artifacts, full data and `.venv` are ignored by Git but remain present locally. Do not load joblib files from unknown sources. The separate organiser test file was not supplied: this app’s test results use the seeded held-out subset of the supplied training file.

## Recorded real-data results

The run uses 162,634 raw CDMs, 162,243 valid-time rows and 13,143 remaining events. Invalid times are excluded; affected known events are also excluded from supervised work. There are 7,962 eligible events with final CDM <1 day and an observation ≥2 days before TCA. Split: 5,573 training / 1,194 validation / 1,195 test events; elevated targets (final logPc ≥−6): 43 / 9 / 9. Critical targets (≥−4): 4 / 1 / 0. Exclusion categories and raw-source SHA256 are in `artifacts/metrics.json`.

The temporal HistGradientBoosting model wins validation MAE, **3.206** versus **3.453** for the latest-fields model. Untouched test MAE is **3.102 log units** (RMSE 5.277). This is a weak probability forecaster: it predicts no elevated events at threshold −6, so test recall is **0/9**, and precision is undefined. At review capacity 50, the selected model catches **0/9**, while existing max-risk signal A catches **3/9**. These results do not establish a safety benefit or improvement over the existing signal. Validation catches 4/9 versus signal A’s 3/9 at K=50; that gain does not generalize to the test subset. Validation false-negative event `2548` is selected programmatically for the replay demonstration.

## Scientific and leakage safeguards

- Only 13 allowed fields enter application processing, plus optional target `risk`. Final targets come from the actual minimum-time row, never a mixture of nonmissing cells or an older label.
- Time runs from larger to smaller days before TCA. A snapshot contains observations with time ≥ its cutoff. Labels and event IDs never enter model features. No observation <2 days enters early forecast features.
- Split events 70/15/15 with seed 42 before augmentation; stratify where supported. Training uses up to three distinct early prefixes weighted to total one per event. Validation and test contain exactly one primary decision row per event. HistGradientBoosting internal row-level holdout/early stopping is disabled.
- Compare latest-fields baseline B and temporal candidate C once on validation MAE; ties prefer B. Selected model stays fitted on training events only. No tuning on test cases.
- Missing and invalid magnitudes become NaN, not safe zeros. Conflicting same-time rows and invalid-time affected events are excluded from supervised labels. Exact duplicates can collapse. Final targets must be finite and ≤0; the observed −30 floor is retained.
- Positive model predictions clip to 0; raw predictions persist. `10**prediction` is a statistical predicted Pc, not an independent covariance-based calculation. `max_risk_estimate` semantics are unverified: signal A is used only for ranking, never physical Pc or MAE.
- Rules: logPc ≥−4 or distance <1 km → CRITICAL; otherwise missing probability/distance → review (both missing → UNVERIFIED); otherwise logPc ≥−6 or distance ≤5 km → ELEVATED; else SAFE. Missing optional material inputs attach review and prevent a nominal SAFE claim. Critical and elevated triggers survive the optional-evidence flag.
- Ops uses the same early decision row for prediction and distance. Without a supported early observation, only explicitly historical screening appears. The <24h count refers to each event’s final relative observation, not simultaneous live warnings.
- Source content fingerprint + model identity governs inference caching and held-out badges. New uploads can be inferred but are external/not evaluated unless their normalized supported content matches exactly. Recorded evaluation panels always identify the original run cohort.
- Replay returns allowed prefix inputs only until reveal; no labels, future rows, outcome-based options or future-based plot bounds. Changing event, source or time clears reveal. Benchmark metrics do not claim calibrated performance at every replay lead time.
- RTN geometry uses supplied relative positions in metres with equal scale and checks their norm against miss distance (tolerance max(10 m, 2%)); discrepancy is shown. It is not an orbit or physical trajectory.

## Interface and validation

Three tabs: **Conjunction Ops** with searchable triage/export/evidence/geometry; **Risk Time Machine** defaulting to verified held-out validation events; **Evaluation & Ops Load** with separate validation/test results, equal-capacity comparisons and full lineage. No synthetic encounters or fabricated metrics appear in the app. Synthetic data are restricted to tests.

Automated tests cover boundaries, label/future mutation invariance, final-row label selection, conflicting chronology, invalid time, ingestion CSV/XLSX/ZIP, leading zeros, source fingerprint, disjoint splits, undefined metric handling, tie-breaking, replay concealment and geometry discrepancy. Real-data smoke checks reloaded prediction parity, identity, export and disjointness. Streamlit AppTest covers source loading, reveal/reset, empty and malformed paths (`.venv/bin/python scripts/app_smoke_test.py`). The HTTP health endpoint returned `ok`. Manual visual browser inspection was unavailable; the UI has been exercised programmatically.

The model is a prototype with sparse positive outcomes and poor rare-risk recall. Full covariance and hard-body radius are absent. Operational validation, calibration and expert review would be needed before consequential use. No fuel, collision-prevention or maneuver-reduction claims are supported.
