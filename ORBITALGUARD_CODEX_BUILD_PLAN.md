# ORBITALGUARD — CODEX IMPLEMENTATION SPECIFICATION

**Challenge:** Autonomous SG 2026 / Airbus Challenge #1 — AI-driven satellite collision avoidance  
**Build window:** 6 hours end-to-end  
**Goal:** A polished, genuinely functional, evidence-driven hackathon submission.  
**Implementation priority:** correctness > convincing measured evidence > one spectacular interactive demo > extra features.

## 0. Instructions to Codex — start here

You are the coding agent implementing this product, not writing another plan. Inspect the repository and attached data first, then create and run the software described below. Work in vertical slices so a functional demonstration exists early. Make reasonable choices without asking for cosmetic clarification. Do not invent predictions, performance scores, model confidence, flight dynamics, or sample data that look real. If full training data is not present, build the fully functional **Sample Exploration Mode**, label ML features as unavailable, and explain exactly where to place `train_data.csv` / `train_data.zip` to enable training. Never pretend the 299-row spreadsheet supports a good critical-risk classifier.

**Implement the MVP sections 1–8 first.** Optional enhancements in section 9 come only after tests, metrics, and the basic demo work. Update README and run tests before calling the project done. If time is constrained, leave out optional features rather than shipping half-built functionality. Treat the task's final artefacts as a working application, a reproducible training script, validation results, automated tests, and a concise pitch script.

**Files currently supplied by organiser/user:**
- `Autonomous SG 2026_Airbus Track.pdf` — source requirements and judging rubric.
- `train_data_sample.xlsx` — 299 CDM rows, 21 events, 103 columns (not the full training set).
- Full dataset may be supplied separately as `train_data.csv`, `train_data.zip`, or files inside an organiser download. Search `data/`, repository root and the attached inputs. Do not assume it exists.

**Useful authoritative references:**
- ESA dataset and fields: https://kelvins.esa.int/collision-avoidance-challenge/data/
- ESA two-day prediction problem: https://kelvins.esa.int/collision-avoidance-challenge/challenge/
- ESA evaluation logic: https://kelvins.esa.int/collision-avoidance-challenge/scoring/
- NASA handbook: https://ntrs.nasa.gov/citations/20205011318

## 1. Product positioning and winning demonstration

**Product name:** OrbitalGuard  
**Tagline:** *See tomorrow's collision risks, today.*  
**Real user:** Satellite conjunction / mission operations analyst who must triage numerous CDMs, see which encounters merit attention, and review evidence before recommending action.

**Core user story:** "Upload historic or new conjunction CDMs, forecast each encounter's eventual collision risk using only information available at least two days before closest approach, rank potential threats, and show how the prediction evolved, what evidence supports it, and where the model fails."

**Signature WOW factor: RISK TIME MACHINE.** An interactive replay of genuine historical events with a prominent **REVEAL OUTCOME** action:
1. Pick an event from a *held-out validation set*, not the training set.
2. Freeze time at an earlier CDM (e.g., TCA −5 days). Future CDMs and the final label remain hidden.
3. Advance through visible historical CDMs until TCA −2 days, showing the available `max_risk_estimate`, miss distance, predicted final `log10(Pc)`, and risk tier. Show an evolving 3D **encounter geometry** display (not fabricated true orbits).
4. Press **Reveal Outcome** to overlay the actual final recorded `risk` and last available miss distance.
5. Display `AI vs baseline`, `detected / missed`, and an evidence panel with features and triggering deterministic rules.
6. Switch to a *real held-out model failure* to prove honesty; disclose where the system fails.

**Secondary WOW factor: 'Ops Load Simulator'.** An interactive slider **"How many encounters can our analysts review?"** (10/25/50/100 or percentages). Compare recall of final high-risk events for AI ranking vs a transparent baseline at identical alert capacity. Show raw counts, numerator/denominator and precision; if there are no positives, display `Not measurable` rather than 100%. This translates ML performance into operational value and strongly addresses the judging rubric.

**IMPORTANT:** These are grounded historical replays and evaluations, not a claim that the demo can autonomously maneuver satellites. Output is decision support, never a flight command.

## 2. Judging rubric — design every screen and metric around it

| Criterion | Weight | What the demo must prove |
|---|---:|---|
| Technical soundness | 25% | Real feature engineering, actual trained model on full data, deterministic rules, working end-to-end ingestion, non-hardcoded inputs. |
| Problem realism | 20% | Named conjunction analyst, clear triage workflow, no unfounded maneuver claims. |
| Honest evaluation | 20% | Disjoint held-out events, baseline comparison, numbers with sample counts, a failure case, zero-positive safeguards. |
| Trust and traceability | 20% | Show the source CDM, event ID, time-to-TCA, applied thresholds, missing evidence and uncertainty limitations. |
| Impact | 10% | Alert burden / recall at fixed review budget vs baseline, measured rather than invented savings. |
| Time discipline | 5% | Fully working, sharply focused ~3-minute pitch and early valid submission. |

## 3. Data specification and non-negotiable scientific correctness

### 3.1 Allowed columns

Airbus explicitly permits the following simplified input fields. Use **only these as model inputs**, including derived features made from their past values:

`event_id`, `time_to_tca`, `miss_distance`, `relative_speed`, `relative_position_r`, `relative_position_t`, `relative_position_n`, `c_object_type`, `c_sigma_r`, `t_sigma_r`, `max_risk_estimate`, `F10`, `AP`.

`event_id` is a grouping key and **must never be treated as a predictive feature**. The supervised target is `risk`, corresponding to `log10(Pc)`, and **MUST NEVER enter the feature matrix, including lagged values**, for the Airbus simplified solution. `risk` may only be used for constructing targets, retrospective replay/outcome reveal, and evaluation. Do not use `mission_id` or other 90 unused columns.

Units: `time_to_tca` is **days**, `miss_distance` and `relative_position_*` are **metres**, `relative_speed` is **m/s**. The observed `max_risk_estimate` values in the sample are negative; treat this as a log-scale estimate until its exact semantics are verified, **never exponentiate or relabel it without a documented transformation**.

### 3.2 Important sample diagnostics (verified)

- 299 rows; 21 event IDs (`0`–`20`); 103 columns.
- 144 rows have `risk == -30` (potential lower-floor/censored values; do not claim exact real probability of 1e-30).
- 26 rows have `-6 <= risk < -4`; 0 have `risk >= -4`.
- 59 rows have `miss_distance < 1000 m`; 48 of those also have `risk < -6`.
- `F10` and `AP` each missing in 34 rows.
- 19/21 encounters have at least one observation with `time_to_tca >= 2`.
- 18/21 encounters have a latest row with `time_to_tca < 1`; 3 end earlier (`event_id` 1,11,20), so final risk is not necessarily close to TCA.
- Final recorded event risk `>= -6`: **0 / 21**. Therefore, the sample cannot support evaluation of rare positive probability-risk events.

### 3.3 Event-level chronology

For each event, sort by `time_to_tca` **descending** (larger time = earlier in the encounter), which is the opposite of ordinary ascending chronology. The latest CDM is the row with the **minimum** `time_to_tca`.

For the primary supervised task:
- Target `y_final`: the `risk` from the latest recorded CDM per event, subject to usable final-record quality criteria.
- For the primary comparable evaluation cohort, require an actual last CDM with `time_to_tca < 1` day; keep excluded counts visible.
- Only allow feature rows with `time_to_tca >= 2` days before TCA. Select the latest eligible row (minimum time-to-TCA among eligible) as the **TCA −2 day decision snapshot**.
- All temporal features at that snapshot must depend only on eligible historical rows **up to and including that snapshot** (never future rows).
- An event without eligible early telemetry is excluded from the early-forecast task; it can still appear in historical exploration with a warning.
- Evaluation has **one prediction per event**, using the latest eligible snapshot. Never count multiple CDMs as independent test cases.
- For model training, optionally create up to 3 chronological early snapshots per *training* event (earliest, middle, latest) to support replay at varying lead times. Weight each snapshot by `1/(snapshots in its event)` so frequently reported events do not dominate. Separate events into train/validation/test **before** making any augmented snapshots. If time is tight, train one snapshot per event and explicitly limit model-based replay to the TCA−2d snapshot.

### 3.4 Holdout and leakage prevention

Split at **event ID**, not row. Preferred for full dataset: stratified event-level train/validation/test split of 70/15/15 by whether final `risk >= -6`, if the positive count permits; otherwise use a seeded grouped split and report its limitations. Zero overlapping event IDs. No model fitting, scaler fitting, imputation fitting, feature selection, threshold tuning, or SHAP-background fitting using test records.

Use `test_data.csv` only for **predictions** if Airbus provides no outcomes for it; do not report test-set accuracy or F2 without labels. Use held-out training events for honest offline evaluation. Record random seed, cohort selection and exclusions.

Verify with an automated test that mutating all rows **after the 2-day cutoff** cannot change a snapshot's predictive features or prediction. Verify that injecting `risk` with arbitrary values in eligible input rows never changes features or predictions.

### 3.5 Validation and upload behavior

- Accept `.csv`, `.xlsx` and optional `.zip` of CSV files; show friendly schema errors.
- Strip completely blank trailing columns; require needed IDs, time and geometry fields; optional features can be NaN.
- Coerce numeric fields safely; preserve `c_object_type` as categorical; display skipped row counts.
- Reject obviously malformed values (negative miss distance, negative time-to-TCA, nonfinite values) or flag records for human review; **never convert invalid data silently into safe outcomes**.
- On uploads without a `risk` column, enable inference/triage but disable outcome reveal and any label-based metrics.
- Cache uploaded/processed data and model inference to maintain responsive UI.

## 4. Proper collision-risk rules

`risk = log10(Pc)`, so `Pc = 10 ** risk` whenever the output has a valid physical/log-probability interpretation.

Airbus challenge-specific thresholds (priority order):

```python
def challenge_tier(predicted_log_pc, miss_distance_m):
    # Return UNVERIFIED, not SAFE, when both are unavailable.
    # If one input is missing, an explicitly high-risk available input can
    # still trigger an alert, but uncertain noncritical tiers must be marked for review.
    if predicted_log_pc is not None and predicted_log_pc >= -4:
        return "CRITICAL"
    if miss_distance_m is not None and miss_distance_m < 1000:
        return "CRITICAL"
    if predicted_log_pc is None or miss_distance_m is None:
        return "REVIEW_REQUIRED"
    if predicted_log_pc >= -6 or miss_distance_m <= 5000:
        return "ELEVATED"
    return "SAFE"
```

- Safe: `logPc < -6` AND miss distance `> 5000 m`.
- Elevated: `-6 <= logPc < -4` OR `1000 <= miss_distance <= 5000 m`, subject to critical priority.
- Critical: `logPc >= -4` OR miss distance `< 1000 m`.
- Exact boundaries should be tested (`-6`, `-4`, `1000`, `5000`).
- Label the distance-only trigger **"critical proximity-screening rule"**, not "confirmed collision danger" or "maneuver is obligatory". A low Pc and short miss distance can coexist.
- If no trained model is available, **do not manufacture `predicted_log_pc`**. Show a distance-based preliminary screen and `AI forecast unavailable`.
- If the only probability value available is `max_risk_estimate`, display it as **existing max-risk estimate**, not actual Pc or learned forecast.
- Show what triggered the alert, the input values used, and the model's source/validity. NASA's recommendation around 1e-4 is an operational mitigation guidance point, but decisions require qualified human review; do not claim the challenge's 1km override is an official NASA physical maneuver rule.

### 4.1 Two distinct risk notions in the UI

1. **Predicted final collision probability**: `Pc_hat = 10 ** model_prediction_logpc` (statistical estimate, not full physics Pc calculation).
2. **Likelihood of elevated final risk**: `P(y_final >= -6 | features)` from a separately trained classifier. This is the model's confidence-like estimated *event outcome frequency*, **not** a physical collision probability. Distinct labels, units, and tooltips. Do not show it at all unless a classifier exists and evaluation supports it.

The dashboard should never imply that the system calculates orbital covariance-based physical Pc independently: the simplified inputs omit data such as full covariance and hard-body radius required for such a computation.

## 5. ML pipeline: minimal, credible, measurable

### 5.1 Baselines (must implement before the fancy model)

- **Baseline A — existing telemetry estimate:** Use the latest eligible `max_risk_estimate` as a log-scale ranking signal, *if its meaning is confirmed*. As a risk forecast proxy label it explicitly as a baseline, not a physical Pc. If its semantics remain unclear, use it for ranking only and compare appropriate ranking metrics.
- **Baseline B — simple tabular model:** A plain HistGradientBoostingRegressor or CatBoostRegressor using ONLY the latest eligible row's allowed features, with minimal preprocessing.
- **Candidate C — temporal model:** CatBoostRegressor or HistGradientBoostingRegressor with latest allowed features and engineered history features.

Feature engineering (only from available prefix): latest value; count of observed CDMs; time span; earliest and latest `miss_distance`; difference / rate of change (with zero-division guards) for `miss_distance`, `max_risk_estimate`, `c_sigma_r`, `t_sigma_r`; min/max/std where well-defined; `log1p` transformed strictly nonnegative magnitudes where useful. Never use `risk` as a feature. Avoid feature explosion; simple models can win.

**Primary deliverable:** predict final `risk` (log10 Pc). Use a regression model; clip physically invalid predictions to a documented range if necessary, and do not hide the fact that `-30` may be a censored/floor value. If the full dataset has adequate positive examples (positive final risk `>= -6`), optionally train a separate binary CatBoostClassifier to rank elevated-risk outcomes. Avoid false certainty and avoid calibrating with test labels.

**Constrained time:** Run baseline first; time-box hyperparameter tuning (no heavyweight grids). Compare the temporal model with baseline on the exact same held-out events, and select the honest best model. Save model artifact and `metrics.json`.

### 5.2 Honest evaluation panel

Calculate where mathematically supported:
- Number of validation events and positive final events (`final risk >= -6`).
- Critical-risk count (`final risk >= -4`) separately; do not imply sample contains any.
- High-risk precision, recall, F2, confusion matrix, false negatives.
- PR-AUC (only if both classes present); no fake zero/one when undefined.
- Regression MAE / RMSE on final log risk and separately for high-risk true events.
- Ranking recall@K for K = 10, 25, 50 or fractional budgets, with transparent numerators and exact baseline comparator.
- Count and characterization of missing/invalid/low-quality inputs.
- At least 1 real failure case, selected programmatically from held-out events (e.g., largest absolute prediction error or false negative).

Whenever the relevant positive count is zero, show `N/A — no positive events in evaluation cohort` rather than 0% or 100% recall. For a subset containing all positives, say explicitly that performance may be biased due to sampling.

**Do not claim** the product saves fuel, prevents collisions, reduces maneuvers by X%, or improves safety by X% unless supported by measurements. The valid first-order impact metric is analyst review workload at a fixed high-risk recall, calculated on held-out data.

## 6. Application design — mission control without fake orbital physics

**Recommended stack for six-hour delivery:** Python 3.11+, Streamlit, pandas, numpy, scikit-learn, CatBoost if installed, Plotly, joblib, pytest. Prefer Plotly/Streamlit over a React+FastAPI split under six hours. If the repo already has a stable React UI and API, keep it only if demonstrably faster than replacing it. `requirements.txt` must be pin-compatible and launchable with one command.

**Visual direction:** elegant, premium aerospace mission control, not generic data science notebook. Deep navy background, cyan/teal active traces, amber warnings, restrained red for critical risk, abundant whitespace, concise copy, strong typography, and readable numbers/units. Avoid unnecessary 3D Earth globes or visual effects that reduce performance.

### Screen A — CONJUNCTION OPS (primary screen)

Top: `ORBITALGUARD | LIVE DATASET MODE: HISTORICAL / SAMPLE / UNLABELED` and a clear source/training status badge.

Hero KPI strip: `Encounters inspected`, `Proximity-screening alerts`, `Predicted elevated/critical`, `Events with TCA <24h` (event-relative, not worldwide concurrent events).

Main area:
- Left: triage table, searchable/sortable by predicted risk tier, model-estimated logPc, distance, TCA, missing-data flag.
- Right: selected event's evidence card with `latest early CDM`, predicted final `logPc`, calculated `Pc` with scientifically correct notation (if modeled), miss distance, applied deterministic rules and "qualified operator review required".
- Below: encounter geometry plot and event history sparkline.

**No fake real-time claim:** these are historical or user-uploaded encounters; `time_to_tca` is relative time, without absolute epoch. A displayed `<24h` count must mean **rows with their own TCA in under 24h**, not simultaneous events in the next 24 clock hours. Never imply the system has up-to-date live orbital tracking.

### Screen B — RISK TIME MACHINE (the demo hero)

Use held-out validation events with clear `held-out` indicator. Controls: event dropdown, timeline scrubber (ordered earliest -> newest), `Play` optional, `Reveal Outcome` toggle/button. By default, hide all future rows and final `risk`. At each early snapshot show:
- `Available at this moment`: time_to_tca, miss distance, max_risk_estimate, position uncertainty, relative speed.
- `AI forecasts final Pc`: regression log10-risk and number in `a × 10^b` format if a trained model is loaded.
- `Why did it change?`: compare current vs previous allowed features, feature contribution if available, and current triggering rules.
- `What actually happened?`: locked until revealed.
- On reveal: final true log-risk, the last CDM time, final distance, AI error and baseline error; call out when final CDM is >1 day from TCA.

**3D wow visual, truthfully labeled:** Plot the target at (0,0,0) and the chaser's *predicted relative position at TCA*, using `relative_position_r/t/n`. Across successive CDMs, plot the revised relative-position points or an uncertainty-informed marker. **Do not connect these points and call them the actual spacecraft trajectory.** Label axes `R / T / N (m)` and caption `Evolution of predicted encounter geometry — not a physical orbital simulation`. Can show a dotted path of *estimates changing*, not objects moving in space. Use responsive Plotly 3D; if it glitches, use 2D scatter/line as stable fallback.

### Screen C — EVIDENCE LAB

- Comparative metrics for baseline vs AI (actual values, n's, cohorts).
- Review-budget slider with Recall@K / precision@K chart and number of critical/high-risk events caught; only use held-out labels.
- Failure case with model prediction, final label, reason for failure/uncertainty, and what the analyst sees.
- Raw source CDM mini-table, feature set and threshold rule explainers.
- Text block "Limits: estimated, not dynamically verified physical Pc; simplified covariance; no automatic maneuvers; evaluation cohort potentially biased".

If full labeled dataset is absent, show `Evaluation unavailable on sample: zero final high-probability events`, a browsable sample replay, and functional deterministic screening. Avoid fabricated charts or demo numbers.

## 7. Repository architecture and API contracts

Suggested compact structure:

```text
orbitalguard/
├── app.py                          # Streamlit application and navigation
├── requirements.txt
├── README.md
├── .streamlit/config.toml         # tasteful dark theme
├── data/
│   ├── train_data_sample.xlsx     # supplied (or symlink/copy from parent)
│   └── README.md                  # where full train_data.csv/zip goes
├── src/
│   ├── __init__.py
│   ├── data.py                     # CSV/XLSX/ZIP import, schema, telemetry ordering
│   ├── features.py                 # strictly causal event snapshot features
│   ├── risk_rules.py               # challenge-specific deterministic decision table
│   ├── models.py                   # training, persistence and inference
│   ├── evaluation.py               # event-level metrics / ranking@K
│   ├── replay.py                   # time-sliced, outcome-hidden histories
│   └── plots.py                    # 3D RTN geometry, risk timeline, metrics
├── scripts/
│   ├── train.py                    # python scripts/train.py --data ...
│   └── smoke_test.py               # quick end-to-end check
├── tests/
│   ├── test_risk_rules.py
│   ├── test_data_features.py
│   ├── test_leakage.py
│   ├── test_replay.py
│   └── test_metrics.py
├── artifacts/                      # generated; not committed
│   ├── model.joblib
│   ├── metrics.json
│   └── validation_predictions.csv
└── PITCH.md                        # <=3-minute spoken script and demo navigation
```

Core Python function interfaces (use clear dataclasses or dictionaries, never hide side effects):

```python
load_cdms(path_or_uploaded_file) -> pd.DataFrame
validate_cdms(df) -> (pd.DataFrame, list[str])
build_event_snapshot(event_df, snapshot_time_to_tca: float) -> dict
build_training_table(df, min_lead_days=2.0, require_final_within_days=1.0) -> (X, y, event_metadata)
classify_challenge_tier(log_pc_or_none, miss_distance_m_or_none) -> RiskDecision
train_models(full_training_df, seed=42) -> TrainedBundle
predict_event(snapshot_feature_dict, model_bundle) -> Prediction
compute_heldout_metrics(validation_predictions) -> dict
get_replay_frame(event_df, snapshot_index, reveal_outcome=False) -> dict
```

`RiskDecision` includes `tier`, `triggered_rules`, `source_fields`, `review_required`. `Prediction` includes `model_version`, `predicted_final_log_pc`, `estimated_final_pc` (if present), and a clear data-quality / model-status field. Avoid invented confidence intervals.

### Commands that must work

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest -q
python scripts/smoke_test.py
python scripts/train.py --data data/train_data.csv --output artifacts/
# or: python scripts/train.py --data data/train_data.zip --output artifacts/
```

Training script must print data counts, train/val/test positive class counts, metrics, artefact paths, warnings, and skipped events. Command-line help should state that `train_data_sample.xlsx` is not sufficient for robust model evaluation.

## 8. Test-driven acceptance criteria — implement these regardless of time

### A. Risk boundaries

| `logPc` | miss distance | Expected |
|---:|---:|---|
| `-7.0` | 6000 m | SAFE |
| `-6.0` | 6000 m | ELEVATED |
| `-5.0` | 6000 m | ELEVATED |
| `-4.0` | 6000 m | CRITICAL |
| `-8.0` | 999 m | CRITICAL (distance-only) |
| `-8.0` | 1000 m | ELEVATED |
| `-8.0` | 5000 m | ELEVATED |
| `-8.0` | 5001 m | SAFE |
| `None` | 9000 m | REVIEW_REQUIRED |
| `None` | 500 m | CRITICAL but missing-probability review flag |
| `-4.2` | `None` | REVIEW_REQUIRED (cannot conclude noncritical) |

### B. Time and leakage

1. `time_to_tca=5, 3, 1` must be displayed in that chronological order.
2. At a 2-day cutoff, `t=1` must never influence feature engineering.
3. Changing labels (`risk`) at any time must not change ML input features.
4. Final `risk` for training is from the latest CDM in each event, not the last Excel row if unsorted.
5. Each held-out event appears only once in primary evaluation.
6. `event_id` never appears among model features.
7. If final risk has no positives, recall/F2/PR-AUC are `N/A`, not 100%.
8. In replay, final outcome and post-snapshot observations cannot be accessed/visible before `Reveal Outcome`.
9. Missing `AP` / `F10` / `c_object_type` handled gracefully, and outcome not silently categorized SAFE.
10. No trained-model inference card appears if there is no actual model artifact.
11. Verify `relative_position_r/t/n` components plausibly reconstruct `miss_distance` (within reasonable rounding), and if not, show discrepancy rather than fake a 3D point.
12. Zip import and Excel sample import both work; CSV export of predictions is correct and reproducible.

### C. Demonstration checks

- Starting from a clean install, an evaluator can launch the app and load the 299-row sample without a traceback.
- With the full labeled dataset added, a model can be trained and evaluated in a reproducible command.
- The UI works with arbitrary `event_id`, not only one scripted example.
- At least one held-out event and one held-out error can be selected; no training outcomes passed off as unseen results.
- Under ordinary laptop conditions, replay and chart changes feel responsive; cache expensive computations.
- The product is explicitly labeled prototype / human-review-only.

## 9. Optional enhancements — only AFTER MVP and tests

Rank in this order:

1. **Polished reveal animation** and visually satisfying two-line AI-vs-baseline outcome overlay. Pure presentation; must not alter metrics.
2. **Causal explanation panel:** show values/trends that changed at each CDM and optional SHAP contributions (label contributions as model attributions, not proven causes).
3. **Data quality sentinel:** flags anomalous miss distance, huge sigmas, missing data, out-of-distribution telemetry; prevent SAFE claims if evidence incomplete.
4. **Conformal/quantile uncertainty** only when genuinely calibrated using a disjoint calibration set and empirically evaluated; otherwise use explicit 'uncertainty not calibrated'.
5. **Model chooser** for baseline vs temporal model, with transparent tradeoffs.

Do **not** build these within six hours: LLM chatbot, autonomous maneuver planner, reinforcement learning, true full-orbit propagator, simulated delta-v, fabricated maneuver effectiveness, network-dependent live satellite tracking, or multiple backend/frontend services without need.

## 10. Six-hour execution schedule and ship gates

| Clock | Deliverable | Release gate |
|---|---|---|
| 0:00–0:25 | Repo scaffolding, load sample, schema/EDA, deterministic rules + tests | All sample rows load and tier boundary tests pass. |
| 0:25–1:30 | Leakage-safe event snapshots and baseline training | Dataset diagnostic report and event-disjoint split. |
| 1:30–2:30 | Regression model and evaluation artefacts (if full training present) | Printed held-out metrics and failure event; no fabricated numbers. |
| 1:15–3:15 | Dashboard built in parallel if teammates available | Conjunction Ops usable with real sample events. |
| 3:15–4:15 | Risk Time Machine with reveal and RTN geometry | Real historical replay with concealed outcomes and honest overlays. |
| 4:15–4:55 | Ops Load Simulator, evidence panel, validations | Review-budget graph works or is explicitly unavailable with no positives. |
| 4:55–5:25 | Run pytest, fresh-start smoke test, fix major bugs | End-to-end demo never crashes. |
| 5:25–6:00 | Freeze app; pitch rehearsal, final outputs, early submission | 3-minute pitch and deploy/run instructions ready. |

**Parallel team split, if available:** Person A owns data/features/training, Person B dashboard/replay, Person C tests/evaluation/insights, Person D integration/pitch and polished interaction. If solo Codex, prioritize vertical slices and ship the baseline dashboard by hour 2.

## 11. Three-minute winning pitch (PITCH.md)

**0:00–0:20 — Real problem.** "Mission operators receive numerous satellite conjunction alerts. The challenge isn't showing another red warning; it's recognizing which encounters deserve attention before the final CDM exists."

**0:20–0:40 — Operational dashboard.** Upload or load the historical data, show triage list, user, thresholds and source CDM. Stress human operator is responsible for decision.

**0:40–1:40 — Risk Time Machine (showpiece).** Select **held-out** event, hide outcome, scrub through genuine early CDMs. Show model forecast and 3D predicted encounter geometry; click reveal; overlay actual final logPc, compare baseline. State results only after real training.

**1:40–2:20 — Measured impact.** Use the analyst-budget slider to show how many truly elevated events are caught at a fixed investigation capacity, AI vs baseline. State numerator, denominator, dataset and split.

**2:20–2:45 — Trust and failure.** Show a real missed case or uncertain prediction, the input evidence, and why this is operator decision support rather than automatic control.

**2:45–3:00 — Close.** "OrbitalGuard turns evolving debris warnings into earlier, explainable, and testable risk triage. We don't hide uncertainty; we make it actionable for the analyst."

## 12. Final output required from Codex

Do not stop after scaffolding. Deliver:

1. Working app and reproducible dependencies.
2. Real-data parser, event-level causal feature engineering, rules and model (if full data exists).
3. `artifacts/metrics.json` and `validation_predictions.csv` when evaluable, with data lineage/counts, model version, positive class support and baseline metrics.
4. Usable risk replay, outcome reveal, evidence panel, geometry chart and workload slider (gracefully disabled when unlabeled/no positive data).
5. Passing tests and automated smoke test, with honest exceptions noted.
6. `README.md`: setup, how to add full data, limits, how predictions are produced, no automatic maneuvers.
7. `PITCH.md`: talk track, exact click path and explicit placeholders for actual measured numbers.
8. Terminal report listing implemented functionality, tests run, available data, actual model/metrics status, known limitations, and how to start the demo.

**Stop conditions:** Never present placeholder metrics as achieved results. If only the provided sample is accessible, ship a beautiful and honest working sample explorer with risk screening and genuine replay; explicitly call the trained AI component blocked pending full training data, rather than silently training on 21 all-negative events.
