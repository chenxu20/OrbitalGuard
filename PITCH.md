# OrbitalGuard — three-minute demonstration

**0:00–0:25 — The operator problem.** Mission analysts receive evolving conjunction messages. OrbitalGuard asks a practical question: using evidence available at least two days before closest approach, which encounters should an analyst inspect? This is historical decision support, not autonomous flight control.

**0:25–0:50 — Conjunction Ops.** Run `.venv/bin/python -m streamlit run app.py`, load the supplied default CSV, and open Conjunction Ops. Show source/model status, the searchable queue, a source CDM and geometry. The distance rule uses the same early observation as the forecast. This is the supplied encounter geometry, not a fabricated spacecraft orbit.

**0:50–1:40 — Risk Time Machine.** Open Risk Time Machine. The default matching-source validation event is `2548`, a real model false negative. This default case has one eligible early observation: show its exact allowed inputs and model forecast, then click Reveal outcome. State the recorded final target and displayed error directly from the screen. For the scrubbing demonstration, choose another event with multiple early observations and advance the slider: only evidence available at that point is visible, and changing time clears the outcome reveal. This case shows why analyst review and measured failure matter.

**1:40–2:25 — Honest assessment.** Open Evaluation & Ops Load, select test and set capacity to 50. Training uses 7,962 eligible events split by event, with only 9 elevated outcomes in the 1,195-event test cohort. Temporal features improve validation MAE from 3.453 to 3.206. Test MAE is 3.102 log units, but rare-event screening is inadequate: no positives at threshold −6; 0 of 9 caught by AI at capacity 50 versus 3 of 9 by the existing max-risk ranking signal. We show this failure clearly instead of turning validation improvement into an unsupported safety claim.

**2:25–2:50 — Engineering evidence.** Show the full report: event-disjoint splits, conservative chronology exclusions, seeded reproducibility, source fingerprints and undefined precision represented honestly. Forecasts are statistical estimates; simplified telemetry cannot independently calculate covariance-based physical Pc. Geometry is checked against reported distance.

**2:50–3:00 — Close.** OrbitalGuard delivers a working analyst workflow and a testable early-warning benchmark. The current model needs better rare-event performance before operational use. Our contribution is making evidence, workload and limitations visible together.
