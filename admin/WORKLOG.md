# WORKLOG.md

## 2026-04-14 — NZ Virology Data Collection (Insha)

**Context:** Team scoped to AUS + NZ focus. Collected NZ respiratory surveillance data to complement Australia NNDSS dataset.

**Work Completed:**
- Built automated scraper in Jupyter Notebook to batch-download weekly virology PDFs from PHF Science (NZ govt public health institute)
- Extracted weekly counts for influenza A/B subtypes, RSV, SARS-CoV-2, rhinovirus, adenovirus, hMPV, parainfluenza, and enterovirus
- Saved to NZ_Airborne_Disease_Data.xlsx with per-year sheets + pivot table (2,728 rows, 2021–2025)
- Files Created: NZ_Airborne_Disease_Scraper.ipynb, NZ_Airborne_Disease_Data.xlsx

**Next Steps:**  Standardize epi-week format, add week_start_date, merge with Australia dataset for R_t modeling

## 2026-04-14 — Scope Update: AUS + NZ Focus (JS)

**Context**: Team decided to focus on Australia + New Zealand, defer Singapore to future work.

**Work Completed**:
- Updated VISION.md, WORKPLAN.md to reflect AUS + NZ scope
- NZ COVID data already exists in WHO_COVID19_cleaned.csv (316 weekly rows)
- NZ-specific respiratory surveillance data collected separately by Insha — 2026-04-14 NZ Virology Data Collection entry

**Next Steps**: Add NZ as Test Set 3 in modeling notebook.

---

## 2026-04-13 — Project Review & Final Planning (JS)

**Context**: Reviewed full project repo to plan final submission tasks.

**Work Completed**:
- Audited all files, folders, and feedback
- Identified missing admin docs (VISION.md, WORKPLAN.md were empty)
- Created VISION.md, updated WORKPLAN.md and WORKLOG.md
- Mapped final rubric requirements to task assignments

**Issues Found**:
- `Code/` folder needs renaming to `work/` per final rubric
- No `submission.md` in `final/` yet
- Professor's checkpoint concern: "leaned heavily on AI" — final report must demonstrate understanding

**Next Steps**: Begin final report, organize notebooks, integrate Singapore test results.

---

## 2026-04-12 — Updated Feature Pipeline Per Professor Feedback (JS)

**Context**: Professor feedback asked for cases_per_100k, lag_5/6, rm2, z12, and regression targets y_2/4/6/8.

**Work Completed**:
- Updated `build_features_for_series()` in code_updated.ipynb:
  - Added `cases_per_100k` normalization using population data
  - Extended lags from 4 to 6
  - Added `roll2_mean` (rm2)
  - Changed z-score from 8-week to 12-week window (z12)
  - Added regression targets `y_2, y_4, y_6, y_8`
- Added regression models (RF, GB, XGB regressors)
- Added regression horizon analysis (MAE vs forecast horizon)
- Updated FEATURE_COLS to include new features
- Files modified: `Code/code_updated.ipynb`

**Impact**: Now addresses both professor's requests — surge classification AND surge size regression with timing tradeoff.

---

## 2026-04-06 — Roshni's V2: Global Training Pipeline (RM)

**Context**: Redesigned modeling approach from Australia-only to global training.

**Work Completed**:
- Restructured pipeline: train on 239 countries, hold out Australia for testing
- Added cross-disease generalization test (COVID-trained model → RSV)
- Added horizon analysis across [1, 2, 3, 4, 6, 8] weeks
- Merged feature engineering + Rt estimation into single function
- Files modified: `Code/code.ipynb` (v2)

**Impact**: Much stronger experimental design — geographic holdout validates global transferability.

---

## 2026-03-28 — Roshni's V1: Australia COVID Pipeline (RM)

**Context**: First complete modeling pipeline for Australia COVID.

**Work Completed**:
- Built feature engineering function with lags, rolling stats, growth rates, z-score, seasonality
- Implemented Rt estimation using log-linear slope method
- Surge labeling: Rt > 1.2 for 3+ consecutive weeks
- Trained 4 models: LogReg, RF, GB, XGBoost with rolling window CV + SMOTE
- 80/20 time-based train/test split on Australia
- Generated `features_covid_aus.csv`
- Files created: `Code/code.ipynb` (v1), `Data/features_covid_aus.csv`

---

## 2026-03-15 — Scope Narrowed Per Professor Feedback (Team)

**Context**: Professor feedback on proposal suggested focusing on specific region and disease.

**Decisions Made**:
- Primary stakeholder: Australian public health agencies
- Focus region: Australia (isolated, clean data available)
- Focus disease: COVID-19 (primary), RSV (cross-disease test)
- Added Rt modeling requirement
- Singapore data retained for external validation

---

## 2026-03-01 — Australia NNDSS Data Collection (RM)

**Context**: Needed Australian disease surveillance data for cross-disease testing.

**Work Completed**:
- Downloaded 40+ NNDSS fortnightly Excel reports from Australian Dept of Health
- Merged into single wide-format file: `AUS_merged_nndss_all_diseases.xlsx`
- Extracted RSV and Measles subsets
- Created AUS data cleaning and biweekly extraction notebooks
- Files created: `Code/AUS_Data_Cleaning.ipynb`, `Code/AUS_Extract_Biweekly.ipynb`

---

## 2026-02-16 — Dataset Refinement (Team)

**Context**: Based on instructor feedback, narrowing scope.

**Work Completed**:
- Researched outbreak datasets for Australia/NZ region
- Reviewed epidemiology literature on R_0 modeling
- Identified NZ data sources: health.govt.nz, flutracking.net

---

## 2026-02-09 — 3 Datasets Identified + Cleaned (Insha)

**Context**: Setting up the “Predicting Pandemic Risk from Disease Surveillance Data” project by finding datasets and cleaning them into a consistent weekly format.

**Work Completed**:
- Collected three datasets: Singapore, Australia, and COVID.
- Cleaned each dataset (fixed data types, handled missing values, removed duplicates).
- Standardized them into the same structure with a weekly date field (week_start_date) so they can be compared/combined.
- [View- Singapore cleaned dataset.ipynb](Singapore%20InfectiousDiseases.ipynb)
- [View- COVID cleaned dataset.ipynb](../Code/WHO_COVID19_Data_Cleaning.ipynb)

**Impact**: All three datasets are now clean and consistent.

**Next Steps**: Combine them into one dataset (if needed) and start baseline modeling + evaluation.

## 2026-02-16 - Dataset Refinement (Insha)
Based on instructor feedback, we're narrowing our project scope to focus on a specific disease (flu, COVID-19, etc.) in an isolated region, preferably Australia or New Zealand. This focused approach will allow us to better incorporate epidemiological factors such as R_0 (basic reproduction number), regional contact rates, transmission modes, and population mobility patterns into our prediction model.

**Steps In Progress**: 
- Currently researching outbreak datasets for our chosen disease/region and reviewing epidemiology literature on R_0 modeling to inform our model development.
- https://www.health.govt.nz/covid-19-novel-coronavirus
- https://info.flutracking.net/reports/new-zealand-reports/
- https://www.phfscience.nz/digital-library/?researchType%5B0%5D=reportItem
