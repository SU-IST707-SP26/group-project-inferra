# WORKLOG.md

## 2026-02-09 — 3 Datasets Identified + Cleaned (Team)

**Context**: Setting up the “Predicting Pandemic Risk from Disease Surveillance Data” project by finding datasets and cleaning them into a consistent weekly format.

**Work Completed**:
- Collected three datasets: Singapore, Australia, and COVID.
- Cleaned each dataset (fixed data types, handled missing values, removed duplicates).
- Standardized them into the same structure with a weekly date field (week_start_date) so they can be compared/combined.
- [View- Singapore cleaned dataset.ipynb](Singapore%20InfectiousDiseases.ipynb)
- [View- COVID cleaned dataset.ipynb](../Code/WHO_COVID19_Data_Cleaning.ipynb)

**Impact**: All three datasets are now clean and consistent.

**Next Steps**: Combine them into one dataset (if needed) and start baseline modeling + evaluation.

## 2026-02-16 - Dataset Refinement (Team)
Based on instructor feedback, we're narrowing our project scope to focus on a specific disease (flu, COVID-19, etc.) in an isolated region, preferably Australia or New Zealand. This focused approach will allow us to better incorporate epidemiological factors such as R_0 (basic reproduction number), regional contact rates, transmission modes, and population mobility patterns into our prediction model.

**Steps In Progress**: 
- Currently researching outbreak datasets for our chosen disease/region and reviewing epidemiology literature on R_0 modeling to inform our model development.
- https://www.health.govt.nz/covid-19-novel-coronavirus
- https://info.flutracking.net/reports/new-zealand-reports/
- https://www.phfscience.nz/digital-library/?researchType%5B0%5D=reportItem
