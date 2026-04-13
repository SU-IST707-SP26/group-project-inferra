# Work Plan

## Active Tasks

### Milestone 1: Data Collection & Cleaning
- ✅ M1.T1 — Collect WHO COVID-19 global weekly data (JS)
- ✅ M1.T2 — Clean WHO COVID data: handle NaN, negatives, type conversion (JS)
- ✅ M1.T3 — Convert WHO COVID data to biweekly format (JS)
- ✅ M1.T4 — Collect and merge Australia NNDSS fortnightly reports (RM)
- ✅ M1.T5 — Extract AUS RSV and Measles datasets from NNDSS (RM)
- ✅ M1.T6 — Collect and clean Singapore infectious disease weekly data (IM)
- ✅ M1.T7 — Obtain population data for cases_per_100k normalization (JS)

### Milestone 2: EDA & Feature Engineering
- ✅ M2.T1 — EDA on Australia COVID timeline and surge patterns (RM)
- ✅ M2.T2 — EDA on NNDSS focus diseases (RSV, Measles, Influenza) (RM)
- ✅ M2.T3 — Build feature engineering pipeline: lags, rolling means, growth rates, z-score, seasonality (RM)
- ✅ M2.T4 — Implement Rt estimation using log-linear slope method (RM)
- ✅ M2.T5 — Define surge labeling logic (Rt > 1.2 for 3+ consecutive weeks) (RM)
- ✅ M2.T6 — Update features per professor feedback: lag_5/6, rm2, z12, cases_per_100k, y_2/4/6/8 (JS)

### Milestone 3: Modeling
- ✅ M3.T1 — Train classification models (LogReg, RF, GB, XGBoost) on global COVID data (RM)
- ✅ M3.T2 — Evaluate on held-out Australia COVID (geographic generalization) (RM)
- ✅ M3.T3 — Evaluate on Australia RSV (cross-disease generalization) (RM)
- ✅ M3.T4 — Run classification horizon analysis (1-8 weeks ahead) (RM)
- ✅ M3.T5 — Add regression models (RF, GB, XGB regressors) for y_2/4/6/8 targets (JS)
- ✅ M3.T6 — Run regression horizon analysis (MAE vs forecast horizon) (JS)
- ❌ M3.T7 — Singapore validation deferred to future work
- ⏳ M3.T8 — Add NZ COVID data as Test Set 3 from existing WHO cleaned data (RM)

### Milestone 4: Dashboard
- ✅ M4.T1 — Build Streamlit dashboard with WHO-style UI (RM)
- ⏳ M4.T2 — Integrate updated model results into dashboard (RM)
- ❌ M4.T3 — Singapore dashboard deferred to future work

### Milestone 5: Presentation (Checkpoint 3)
- [ ] M5.T1 — Create 7-8 slide presentation per rubric (Team)
- [ ] M5.T2 — Rehearse presentation (Team)

### Milestone 6: Final Report & Submission
- ✅ M6.T1 — Rename Code/ folder to work/ per rubric requirements (JS)
- ✅ M6.T2 — Rename and organize notebooks with clear names (JS)
- ⏳ M6.T3 — Write submission.md in final/ folder with all required sections (Team)
  - ⏳ M6.T3a — Title, Team, Introduction (JS)
  - ⏳ M6.T3b — Literature Review (IM)
  - ⏳ M6.T3c — Data section with 2-5 visualizations (RM)
  - ⏳ M6.T3d — Methods section: features, Rt, models, horizons (Team)
  - ⏳ M6.T3e — Results section with metrics, tables, charts (Team)
  - ⏳ M6.T3f — Discussion: stakeholder connection (JS)
  - ⏳ M6.T3g — Limitations and Future Work (IM)
- ✅ M6.T4 — Create supporting files index in report (JS)
- ✅ M6.T5 — Remove data files from repo, provide download links instead (JS)
- [ ] M6.T6 — Final review and submission (Team)

## Changelog

### 2026-04-13
- (JS) Created WORKPLAN.md with full milestone structure
- (JS) Backfilled completed tasks from M1-M3
- (JS) Added M6 (Final Report) tasks based on rubric requirements

### 2026-03-15
- (Team) Narrowed stakeholder to Australian public health agencies per professor feedback
- (Team) Added Rt modeling and regression targets per professor feedback

### 2026-02-09
- (Team) Initial project setup: 3 datasets identified and cleaned

