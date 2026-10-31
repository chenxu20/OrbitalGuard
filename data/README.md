# Data inputs

The supplied `train_data.csv` is a real organiser dataset (~233 MB); it is intentionally ignored by Git. Put CSV/XLSX/ZIP files here or select another local path in the app. ZIP files must contain exactly one CSV to prevent ambiguous overlapping event IDs. No archive files are extracted.

Required headers: `event_id,time_to_tca,miss_distance`. Optional supported headers: `relative_speed,relative_position_r,relative_position_t,relative_position_n,t_sigma_r,c_sigma_r,c_object_type,max_risk_estimate,F10,AP` and label `risk`. Unsupported fields are ignored. Missing optional evidence is flagged; missing position disables geometry. `risk = log10(Pc)` and final targets must be ≤0.

No separate organiser test file or sample spreadsheet was supplied in this workspace. Assessment uses an event-disjoint held-out subset. Small or unlabeled files support exploration but do not establish reliable model performance. Preserve the real data’s use conditions when sharing.
